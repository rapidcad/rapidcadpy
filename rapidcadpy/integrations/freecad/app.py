"""
FreeCAD App – top-level document manager and App implementation.
"""

from typing import Optional, Tuple, Union

from ...app import App

VectorLike = Union[Tuple[float, float, float], Tuple[float, float]]


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
        import FreeCAD

        self._fc_doc = FreeCAD.newDocument(doc_name)

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Bulk export helpers
    # ------------------------------------------------------------------

    def to_step(self, file_name: str) -> None:
        """Export all registered shapes to a single STEP file."""
        if not self._shapes:
            raise ValueError("No shapes to export")
        import Part

        objs = [s.obj for s in self._shapes if hasattr(s, "obj")]
        Part.export(objs, file_name)

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

    def to_fcstd(self, file_name: str) -> None:
        """
        Save the document in FreeCAD's native .FCStd format.

        Each registered shape is added to the document as a named
        ``Part::Feature`` object so that the full shape tree is visible
        and editable when the file is reopened in FreeCAD.  The document
        is then serialised with ``Document.saveAs()``, which writes a
        compressed archive containing the shape BRep data and the XML
        model description – preserving the sequence of operations as
        separate features in the tree.

        Args:
            file_name: Destination path (should end in ``.FCStd``).
        """
        import FreeCAD

        doc = self._fc_doc

        for i, shape in enumerate(self._shapes):
            if not hasattr(shape, "obj"):
                continue
            feature_name = f"Shape{i}"
            # Reuse an existing feature of the same name if present (idempotent
            # on repeated calls), otherwise create a new one.
            if feature_name in [o.Name for o in doc.Objects]:
                feature = doc.getObject(feature_name)
            else:
                feature = doc.addObject("Part::Feature", feature_name)
            feature.Shape = shape.obj

        doc.recompute()
        doc.saveAs(file_name)
