"""
FreeCAD Shape – wraps a Part.Shape and implements the Shape ABC.
"""

import os
import tempfile
from typing import List, Optional, Union, Any

from ...shape import Shape


class FreeCADShape(Shape):
    """
    Concrete Shape backed by a FreeCAD ``Part.Shape`` object stored in
    ``self.obj``. Maintains a feature tree in the FreeCAD document.

    self.obj: The computed OCC shape (for Python operations)
    self._doc: Reference to the FreeCAD document (to add features)
    self._current_feature: The Part::Feature or derived object that holds this shape
    """

    def __init__(self, obj, app, doc=None, current_feature=None) -> None:  # type: ignore[override]
        super().__init__(obj, app)
        self._doc = doc
        self._current_feature = current_feature
        self._feature_counter = 0

    def _doc_get_next_index(self) -> int:
        """Get next feature index for naming."""
        self._feature_counter += 1
        return self._feature_counter

    def _store_result_shape(self, result_shape, prefix: str) -> None:
        """Update ``self.obj`` and mirror the result into the FreeCAD document."""
        self.obj = result_shape
        if self._doc and self._current_feature:
            feature = self._doc.addObject(
                "Part::Feature", f"{prefix}_{self._doc_get_next_index()}"
            )
            feature.Shape = result_shape
            self._doc.recompute()
            self._current_feature = feature

    def _raw_edges(self) -> List[Any]:
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
        return float(self.obj.Volume)

    def to_stl(self, file_name: str) -> None:
        """Export shape to an ASCII or binary STL file."""
        self.obj.exportStl(file_name)

    def to_fcstd(self, file_name: str) -> None:
        """Export this shape to a FreeCAD .FCStd file.

        If this shape was created with a document, the entire feature tree
        (Pad, Boolean, Fillet, etc.) is saved. Otherwise, just the geometry
        is realized as a Part::Feature.
        """
        import FreeCAD

        if self._doc:
            # Save the existing feature tree
            self._doc.saveAs(file_name)
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
        pyvista, vedo.
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
        """Boolean subtraction – modifies this shape and mirrors the result into the doc."""
        self._store_result_shape(self.obj.cut(other.obj), "Cut")
        self._clear_edge_selection()
        return self

    def union(
        self, other: Union["FreeCADShape", List["FreeCADShape"]]
    ) -> "FreeCADShape":
        """Boolean union – modifies this shape and mirrors the result into the doc."""
        others = [other] if not isinstance(other, list) else other

        for s in others:
            self._store_result_shape(self.obj.fuse(s.obj), "Fuse")

        self._clear_edge_selection()
        return self

    # ------------------------------------------------------------------
    # Extra operations (not abstract but useful)
    # ------------------------------------------------------------------

    def translate(
        self, x: float = 0.0, y: float = 0.0, z: float = 0.0
    ) -> "FreeCADShape":
        """Translate the shape in-place by (x, y, z)."""
        import FreeCAD

        vec = FreeCAD.Vector(x, y, z)
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
    backends = ["pyvista", "vedo"] if backend == "auto" else [backend]
    last_error: Optional[Exception] = None

    for name in backends:
        try:
            if name == "pyvista":
                _render_pyvista(stl_file, output, view, width, height)
                return
            elif name == "vedo":
                _render_vedo(stl_file, output, view, width, height)
                return
        except ImportError as exc:
            last_error = exc
            continue

    raise ImportError(
        "No rendering backend available. Install pyvista or vedo. "
        f"Last error: {last_error}"
    )


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
