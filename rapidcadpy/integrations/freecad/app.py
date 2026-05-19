"""
FreeCAD App – top-level document manager and App implementation.
"""

import os
import sys
from typing import Any, Optional, Tuple, Union

from ...app import App
from ...primitives import Arc, Circle, Line

VectorLike = Union[Tuple[float, float, float], Tuple[float, float]]


def ensure_freecad_python_path() -> str | None:
    """Add FreeCAD module directory to ``sys.path`` when discoverable."""
    candidates = [
        os.environ.get("FREECAD_LIB_PATH", "").strip(),
        "/Applications/FreeCAD.app/Contents/Resources/lib",
        "/usr/lib/freecad/lib",
        "/usr/lib64/freecad/lib",
    ]

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue

        normalized = os.path.abspath(candidate)
        if normalized in seen or not os.path.exists(normalized):
            continue
        seen.add(normalized)

        os.environ.setdefault("FREECAD_LIB_PATH", normalized)
        if normalized not in sys.path:
            sys.path.insert(0, normalized)
        return normalized

    return None


class FreeCADApp(App):
    """
    FreeCAD implementation of the App base class.

    Creates and owns a headless FreeCAD document.  All shapes live as
    Part.Shape objects (stored in FreeCADShape.obj) and are independent of
    the document tree – the document is kept mostly as a namespace so that
    FreeCAD's internal bookkeeping stays happy in headless mode.
    """

    def __init__(
        self,
        doc_name: str = "RapidCADPy_Doc",
        silent_geometry_failures: bool = False,
    ):
        super().__init__(silent_geometry_failures=silent_geometry_failures)
        ensure_freecad_python_path()
        import FreeCAD

        self._fc_doc = FreeCAD.newDocument(doc_name)
        self._feature_counter = 0

    def get_doc(self):
        """Get the underlying FreeCAD document."""
        return self._fc_doc

    def get_next_feature_index(self) -> int:
        """Get next feature index for naming."""
        self._feature_counter += 1
        return self._feature_counter

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def sketch_3d(self):
        """Entry point for building 3D path sketches (wires)."""
        from .sketch3d import FreeCADSketch3D

        return FreeCADSketch3D(self)

    @property
    def workplane_class(self):
        from .workplane import FreeCADWorkplane

        return FreeCADWorkplane

    @property
    def sketch_class(self):
        from .sketch2d import FreeCADSketch2D

        return FreeCADSketch2D

    # ------------------------------------------------------------------
    # Document helpers
    # ------------------------------------------------------------------

    def work_plane(
        self,
        name: str = "XY",
        offset: Optional[float] = None,
        origin: Optional[VectorLike] = None,
        normal: Optional[VectorLike] = None,
    ):
        """Create a workplane by name/offset or by absolute origin + normal."""
        if origin is not None and normal is not None:
            return self.workplane_class.from_origin_normal(
                app=self, origin=origin, normal=normal
            )
        return super().work_plane(name=name, offset=offset)

    def reset(self) -> None:
        """Close the current document and open a fresh one."""
        import FreeCAD

        FreeCAD.closeDocument(self._fc_doc.Name)
        self._fc_doc = FreeCAD.newDocument("RapidCADPy_Doc")
        self._shapes.clear()
        self._workplanes.clear()

    def _make_feature_name(self, prefix: str, index: int) -> str:
        return f"{prefix}{index}"

    # ------------------------------------------------------------------
    # Bulk export helpers
    # ------------------------------------------------------------------

    def _upsert_doc_shapes(self, shapes, feature_name_prefix: str = "Shape"):
        doc = self._fc_doc
        existing = {obj.Name: obj for obj in doc.Objects}
        features = []

        for i, shape in enumerate(shapes):
            if not hasattr(shape, "obj"):
                continue

            feature_name = (
                feature_name_prefix if len(shapes) == 1 else f"{feature_name_prefix}{i}"
            )
            feature = existing.get(feature_name)
            if feature is None:
                feature = doc.addObject("Part::Feature", feature_name)
            feature.Label = feature_name
            feature.Shape = shape.obj
            features.append(feature)

        if not features:
            raise ValueError("No valid shapes to export")

        doc.recompute()
        return features

    def to_step(self, file_name: str) -> None:
        """Export all registered shapes to a single STEP file."""
        if not self._shapes:
            raise ValueError("No shapes to export")

        if len(self._shapes) == 1:
            self._shapes[0].to_step(file_name)
            return

        import Part

        features = self._upsert_doc_shapes(self._shapes, feature_name_prefix="Shape")
        Part.export(features, file_name)

    def to_stl(self, file_name: str) -> None:
        """Export all registered shapes to a single STL file."""
        if not self._shapes:
            raise ValueError("No shapes to export")

        if len(self._shapes) == 1:
            self._shapes[0].to_stl(file_name)
        else:
            # Fuse all shapes into one compound and export
            combined = self._shapes[0].obj
            for s in self._shapes[1:]:
                if hasattr(s, "obj"):
                    combined = combined.fuse(s.obj)
            combined.exportStl(file_name)

    def to_fcstd(
        self,
        file_name: str,
        shapes=None,
        feature_name_prefix: str = "Shape",
    ) -> None:
        """Save the document in FreeCAD's native .FCStd format.

        If a single shape with its own document is provided, saves that document's
        feature tree. Otherwise, writes all registered shapes as Part::Feature objects.
        Geometry is eager — all features are realized immediately as the
        shapes are built.

        Args:
            file_name: Destination path (should end in ``.FCStd``).
            shapes: Optional explicit shapes to write instead of the full app registry.
            feature_name_prefix: Base object name to use in the document tree.
        """
        target_shapes = self._shapes if shapes is None else shapes

        # If single shape with its own doc, save that directly
        if (
            len(target_shapes) == 1
            and hasattr(target_shapes[0], "_doc")
            and target_shapes[0]._doc
        ):
            target_shapes[0]._doc.saveAs(file_name)
            return

        # Otherwise, add shapes to app doc
        self._upsert_doc_shapes(target_shapes, feature_name_prefix=feature_name_prefix)
        self._fc_doc.saveAs(file_name)
