"""
FreeCAD Shape – wraps a Part.Shape and implements the Shape ABC.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union, Any

from ...cad_objects import CadDocument, CadFeature
from ...shape import Shape
from .cad_adapter import FreeCADAdapter


class FreeCADShape(Shape):
    """
    Concrete Shape backed by a FreeCAD ``Part.Shape`` object stored in
    ``self.obj``. Maintains a feature tree in the FreeCAD document.

    self.obj: The computed OCC shape (for Python operations)
    ``document`` and ``feature`` publicly expose backend-neutral references to
    the live FreeCAD document and feature.
    """

    def __init__(
        self,
        obj,
        app,
        doc=None,
        current_feature=None,
        *,
        document: Optional[CadDocument] = None,
        feature: Optional[CadFeature] = None,
    ) -> None:  # type: ignore[override]
        cad_document = document or (feature.document if feature is not None else None)
        if cad_document is None:
            cad_document = self._resolve_document(app, doc)
        cad_feature = feature or self._wrap_feature(cad_document, current_feature)
        super().__init__(
            obj,
            app,
            document=cad_document,
            feature=cad_feature,
        )
        self._feature_counter = 0

    @staticmethod
    def _resolve_document(app, native_document) -> Optional[CadDocument]:
        if native_document is None:
            return None
        if app is not None and hasattr(app, "get_cad_document"):
            cad_document = app.get_cad_document()
            if cad_document.native_handle is native_document:
                return cad_document
        adapter = FreeCADAdapter()
        return CadDocument(
            backend=adapter.backend_name,
            native_handle=native_document,
            adapter=adapter,
            name=str(getattr(native_document, "Name", "")),
            label=str(getattr(native_document, "Label", "")),
            file_name=str(getattr(native_document, "FileName", "")),
        )

    @staticmethod
    def _wrap_feature(
        document: Optional[CadDocument],
        native_feature,
        preferred_id: Optional[str] = None,
    ) -> Optional[CadFeature]:
        if document is None or native_feature is None:
            return None
        native_name = str(getattr(native_feature, "Name", ""))
        native_type = str(getattr(native_feature, "TypeId", ""))
        if preferred_id is None:
            rapidcad_id = document.object_id(native_name)
        else:
            rapidcad_id = document.bind_object_id(native_name, preferred_id)
        capabilities = {"geometry", "get_properties", "set_properties"}
        if native_type not in {"Part::Feature", "PartDesign::Feature"}:
            capabilities.add("feature_history")
        return CadFeature(
            id=rapidcad_id,
            document=document,
            native_handle=native_feature,
            native_name=native_name,
            native_type=native_type,
            label=str(getattr(native_feature, "Label", native_name)),
            capabilities=frozenset(capabilities),
        )

    def _bind_feature(self, native_feature) -> None:
        preferred_id = self.feature.id if self.feature is not None else None
        self.bind_native(
            self.document,
            self._wrap_feature(
                self.document,
                native_feature,
                preferred_id=preferred_id,
            ),
        )

    @property
    def _doc(self):
        """Deprecated raw FreeCAD document alias; use ``document``."""
        return self.document.native_handle if self.document is not None else None

    @property
    def _current_feature(self):
        """Deprecated raw FreeCAD feature alias; use ``feature``."""
        return self.feature.native_handle if self.feature is not None else None

    def _doc_get_next_index(self) -> int:
        """Get next feature index for naming."""
        self._feature_counter += 1
        return self._feature_counter

    def _store_result_shape(self, result_shape, prefix: str) -> None:
        """Update ``self.obj`` and mirror the result into the FreeCAD document."""
        self.obj = result_shape
        if self.document is not None and self.feature is not None:
            native_document = self.document.native_handle
            feature = native_document.addObject(
                "Part::Feature", f"{prefix}_{self._doc_get_next_index()}"
            )
            feature.Shape = result_shape
            self.document.recompute()
            self._bind_feature(feature)

    def _make_boolean_feature(
        self, type_id: str, others: List["FreeCADShape"], prefix: str
    ) -> bool:
        """Create a parametric FreeCAD boolean referencing operand features.

        ``Part::Cut`` / ``Part::MultiFuse`` / ``Part::MultiCommon`` reference the
        operand document objects and let FreeCAD recompute the result, so the
        FCStd export keeps an editable boolean node (with the operand sketches /
        extrusions nested underneath) instead of a baked solid.

        Returns ``True`` when the parametric feature was created; ``False`` when
        an operand is not document-backed (caller should fall back to a baked
        OCC boolean so geometry is still correct).
        """
        if self.document is None or self.feature is None:
            return False

        operand_feats = []
        for s in others:
            if (
                s.feature is None
                or s.document is None
                or s.document.native_handle is not self.document.native_handle
            ):
                return False
            operand_feats.append(s.feature.native_handle)

        doc = self.document.native_handle
        boolean = doc.addObject(type_id, f"{prefix}_{self._doc_get_next_index()}")
        if type_id == "Part::Cut":
            boolean.Base = self.feature.native_handle
            boolean.Tool = operand_feats[0]
        else:  # Part::MultiFuse / Part::MultiCommon
            boolean.Shapes = [self.feature.native_handle] + operand_feats
        self.document.recompute()

        self.obj = boolean.Shape
        self._bind_feature(boolean)
        return True

    def _raw_edges(self) -> List[Any]:
        self.refresh_from_feature()
        return list(self.obj.Edges)

    def _is_linear_edge(self, edge) -> bool:
        return len(edge.Vertexes) >= 2

    def _edge_direction_vector(self, edge) -> tuple:
        start = edge.Vertexes[0].Point
        end = edge.Vertexes[-1].Point
        return (
            float(end.x - start.x),
            float(end.y - start.y),
            float(end.z - start.z),
        )

    def _apply_fillet_to_edges(
        self, edges: List[Any], radius: float, selector: Optional[str] = None
    ) -> None:
        # Compute the filleted shape
        try:
            filleted = self.obj.makeFillet(float(radius), edges)
        except Exception:
            # Per-edge fallback
            filleted = self.obj
            for edge in edges:
                try:
                    filleted = filleted.makeFillet(float(radius), [edge])
                except Exception:
                    continue

        self._store_result_shape(filleted, "Fillet")

    # ------------------------------------------------------------------
    # Abstract implementations
    # ------------------------------------------------------------------

    def volume(self) -> float:
        """Return the volume of the shape (model units³)."""
        self.refresh_from_feature()
        return float(self.obj.Volume)

    def to_stl(self, file_name: str) -> None:
        """Export shape to an ASCII or binary STL file."""
        self.refresh_from_feature()
        self.obj.exportStl(file_name)

    def to_fcstd(self, file_name: str) -> None:
        """Export this shape to a FreeCAD .FCStd file.

        If this shape was created with a document, the entire feature tree
        (Pad, Boolean, Fillet, etc.) is saved. Otherwise, just the geometry
        is realized as a Part::Feature.
        """
        import FreeCAD

        if self.document is not None:
            # Save the existing feature tree
            self.document.save(file_name)
        else:
            # No doc: create a minimal one with just this shape
            doc = FreeCAD.newDocument("export")
            feature = doc.addObject("Part::Feature", "Shape")
            feature.Shape = self.obj
            doc.recompute()
            doc.saveAs(file_name)
            FreeCAD.closeDocument(doc.Name)

    def to_step(self, file_name: str) -> None:
        """Export shape to a STEP file."""
        self.refresh_from_feature()
        self.obj.exportStep(file_name)

    def to_png(
        self,
        file_name: Optional[str] = None,
        view: str = "iso",
        width: int = 800,
        height: int = 600,
        backend: str = "auto",
    ) -> None:
        """
        Render the shape to a PNG via a temporary STL export.

        Supported rendering back-ends (tried in order when backend='auto'):
        pillow (software/headless), pyvista, vedo.
        """
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            tmp_stl = tmp.name

        try:
            self.to_stl(tmp_stl)
            _render_stl(tmp_stl, file_name, view, width, height, backend)
        finally:
            if os.path.exists(tmp_stl):
                os.remove(tmp_stl)

    def cut(self, other: "FreeCADShape") -> "FreeCADShape":
        """Boolean subtraction as a parametric ``Part::Cut`` (baked fallback)."""
        if not self._make_boolean_feature("Part::Cut", [other], "Cut"):
            self.refresh_from_feature()
            other.refresh_from_feature()
            self._store_result_shape(self.obj.cut(other.obj), "Cut")
        self._clear_edge_selection()
        return self

    def union(
        self, other: Union["FreeCADShape", List["FreeCADShape"]]
    ) -> "FreeCADShape":
        """Boolean union as a parametric ``Part::MultiFuse`` (baked fallback)."""
        others = [other] if not isinstance(other, list) else other

        if not self._make_boolean_feature("Part::MultiFuse", others, "Fuse"):
            self.refresh_from_feature()
            for s in others:
                s.refresh_from_feature()
                self._store_result_shape(self.obj.fuse(s.obj), "Fuse")

        self._clear_edge_selection()
        return self

    # ------------------------------------------------------------------
    # Extra operations (not abstract but useful)
    # ------------------------------------------------------------------

    def translate(
        self, x: float = 0.0, y: float = 0.0, z: float = 0.0
    ) -> "FreeCADShape":
        """Translate the shape in-place by (x, y, z).

        When the shape is document-backed, the feature's ``Placement`` is moved
        (and ``obj`` re-synced from it) so the document feature stays consistent
        with ``obj`` — required for parametric booleans that reference it. Falls
        back to a direct OCC translate otherwise.
        """
        import FreeCAD

        vec = FreeCAD.Vector(x, y, z)
        if self.document is not None and self.feature is not None:
            feature = self.feature.native_handle
            feature.Placement = FreeCAD.Placement(vec, FreeCAD.Rotation()).multiply(
                feature.Placement
            )
            self.document.recompute()
            self.obj = feature.Shape
        else:
            self.obj.translate(vec)
        self._clear_edge_selection()
        return self


# ---------------------------------------------------------------------------
# Internal rendering helpers (shared with sketch2d)
# ---------------------------------------------------------------------------


def _render_stl(
    stl_file: str,
    output: Optional[str],
    view: str,
    width: int,
    height: int,
    backend: str = "auto",
) -> None:
    """Render *stl_file* to *output* (or display interactively if None)."""
    if backend == "auto":
        backends = ["pillow", "pyvista", "vedo"] if output else ["pyvista", "vedo"]
    else:
        backends = [backend]
    last_error: Optional[Exception] = None

    for name in backends:
        try:
            if name == "pillow":
                _render_pillow(stl_file, output, view, width, height)
                return
            elif name == "pyvista":
                _render_pyvista(stl_file, output, view, width, height)
                return
            elif name == "vedo":
                _render_vedo(stl_file, output, view, width, height)
                return
        except ImportError as exc:
            last_error = exc
            continue

    raise ImportError(
        "No rendering backend available. Install Pillow, pyvista, or vedo. "
        f"Last error: {last_error}"
    )


def _load_stl_triangles(stl_file: str):
    """Load binary or ASCII STL triangles into an ``(n, 3, 3)`` array."""
    import numpy as np

    data = Path(stl_file).read_bytes()
    if len(data) >= 84:
        triangle_count = int.from_bytes(data[80:84], "little")
        expected_size = 84 + triangle_count * 50
        if triangle_count > 0 and expected_size == len(data):
            record_type = np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attribute", "<u2"),
                ]
            )
            records = np.frombuffer(
                data, dtype=record_type, count=triangle_count, offset=84
            )
            return records["vertices"].astype(float, copy=True)

    vertices = []
    for line in data.decode("ascii", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(value) for value in parts[1:]])
    if not vertices or len(vertices) % 3:
        raise ValueError(f"Could not read STL triangles from {stl_file}")
    return np.asarray(vertices, dtype=float).reshape((-1, 3, 3))


def _render_pillow(
    stl_file: str, output: Optional[str], view: str, width: int, height: int
) -> None:
    """Render STL triangles with Pillow only (no GUI or OpenGL required)."""
    if not output:
        raise ValueError("The Pillow renderer requires an output path.")

    import numpy as np
    from PIL import Image, ImageDraw

    triangles = _load_stl_triangles(stl_file)
    points = triangles.reshape((-1, 3))
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    triangles = triangles - center

    view_name = view.lower()
    if view_name in ("iso", "isometric"):
        camera = np.asarray((1.0, -1.0, 0.8))
        up_hint = np.asarray((0.0, 0.0, 1.0))
    elif view_name in ("front", "y"):
        camera = np.asarray((0.0, -1.0, 0.0))
        up_hint = np.asarray((0.0, 0.0, 1.0))
    elif view_name in ("top", "z"):
        camera = np.asarray((0.0, 0.0, 1.0))
        up_hint = np.asarray((0.0, 1.0, 0.0))
    elif view_name in ("right", "x"):
        camera = np.asarray((1.0, 0.0, 0.0))
        up_hint = np.asarray((0.0, 0.0, 1.0))
    else:
        raise ValueError(f"Unsupported view '{view}'. Use iso, front, top, or right.")

    camera = camera / np.linalg.norm(camera)
    screen_right = np.cross(camera, up_hint)
    screen_right = screen_right / np.linalg.norm(screen_right)
    screen_up = np.cross(screen_right, camera)

    projected_x = triangles @ screen_right
    projected_y = triangles @ screen_up
    depth = triangles @ camera
    x_min, x_max = float(projected_x.min()), float(projected_x.max())
    y_min, y_max = float(projected_y.min()), float(projected_y.max())
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    margin = 0.08
    scale = min(
        width * (1.0 - 2.0 * margin) / x_span,
        height * (1.0 - 2.0 * margin) / y_span,
    )
    x_offset = (width - x_span * scale) / 2.0 - x_min * scale
    y_offset = (height - y_span * scale) / 2.0 + y_max * scale

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    face_vectors_1 = triangles[:, 1] - triangles[:, 0]
    face_vectors_2 = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(face_vectors_1, face_vectors_2)
    normal_lengths = np.linalg.norm(normals, axis=1)
    normal_lengths[normal_lengths == 0] = 1.0
    normals = normals / normal_lengths[:, None]
    light = np.asarray((0.4, -0.6, 1.0))
    light = light / np.linalg.norm(light)
    brightness = 0.35 + 0.65 * np.abs(normals @ light)

    # Cull back-facing triangles, then draw far faces before near faces.
    visible = np.flatnonzero((normals @ camera) > 1e-8)
    if not len(visible):
        visible = np.arange(len(triangles))
    draw_order = visible[np.argsort(depth.mean(axis=1)[visible])]
    for index in draw_order:
        polygon = [
            (
                float(projected_x[index, vertex] * scale + x_offset),
                float(y_offset - projected_y[index, vertex] * scale),
            )
            for vertex in range(3)
        ]
        shade = int(105 + 125 * float(brightness[index]))
        draw.polygon(polygon, fill=(shade, shade, shade))

    image.save(output, format="PNG")


def _render_pyvista(
    stl_file: str, output: Optional[str], view: str, width: int, height: int
) -> None:
    import pyvista as pv

    if output:
        pv.OFF_SCREEN = True

    mesh = pv.read(stl_file)
    plotter = pv.Plotter(off_screen=(output is not None), window_size=[width, height])
    plotter.add_mesh(mesh, color="lightgray", show_edges=False)
    plotter.background_color = "white"

    view = view.lower()
    if view in ("iso", "isometric"):
        plotter.camera_position = "iso"
    elif view in ("front", "y"):
        plotter.camera.position = (0, -10, 0)
        plotter.camera.focal_point = (0, 0, 0)
        plotter.camera.up = (0, 0, 1)
    elif view in ("top", "z"):
        plotter.camera.position = (0, 0, 10)
        plotter.camera.focal_point = (0, 0, 0)
        plotter.camera.up = (0, 1, 0)
    elif view in ("right", "x"):
        plotter.camera.position = (10, 0, 0)
        plotter.camera.focal_point = (0, 0, 0)
        plotter.camera.up = (0, 0, 1)

    if output:
        plotter.show(screenshot=output, auto_close=True)
    else:
        plotter.show()


def _render_vedo(
    stl_file: str, output: Optional[str], view: str, width: int, height: int
) -> None:
    import vedo

    mesh = vedo.load(stl_file)
    view = view.lower()

    cam_kwargs: dict = {}
    if view in ("iso", "isometric"):
        cam_kwargs = {"azimuth": 45, "elevation": 30}
    elif view in ("front", "y"):
        cam_kwargs = {"azimuth": 0, "elevation": 0}
    elif view in ("top", "z"):
        cam_kwargs = {"azimuth": 0, "elevation": 90}
    elif view in ("right", "x"):
        cam_kwargs = {"azimuth": 90, "elevation": 0}

    if output:
        plotter = vedo.Plotter(offscreen=True, size=(width, height))
        plotter.show(mesh, viewup="z")
        if cam_kwargs:
            plotter.camera.Azimuth(cam_kwargs.get("azimuth", 0))
            plotter.camera.Elevation(cam_kwargs.get("elevation", 0))
        plotter.screenshot(output)
        plotter.close()
    else:
        plotter = vedo.Plotter(size=(width, height))
        plotter.show(mesh, viewup="z", camera=cam_kwargs if cam_kwargs else None)
