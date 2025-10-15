from typing import TYPE_CHECKING, Optional, Any

from rapidcadpy.app import App
from rapidcadpy.cad_types import VectorLike

if TYPE_CHECKING:
    from rapidcadpy.integrations.inventor.workplane import InventorWorkPlane


class InventorApp(App):
    def __init__(self, headless: bool = False):
        try:
            import win32com.client as win32
            from win32com.client import gencache, Dispatch
        except ImportError:
            raise ImportError("pywin32 is required for Inventor integration. Install with: pip install pywin32")

        from rapidcadpy.integrations.inventor.workplane import InventorWorkPlane

        super().__init__(InventorWorkPlane)
        try:
            self.inventor_app = win32.GetActiveObject("Inventor.Application")
        except Exception:
            self.inventor_app = Dispatch("Inventor.Application")
            self.inventor_app.Visible = not headless

        # Load the Inventor COM wrapper (type lib)
        self.mod = gencache.EnsureModule(
            "{D98A091D-3A0F-4C3E-B36E-61F62068D488}", 0, 1, 0   
        )

        self.transient_geom = self.inventor_app.TransientGeometry
        self.transient_obj = self.inventor_app.TransientObjects


    def new_document(self):
        import win32com.client as win32

        # Create a new PartDocument
        self.inventor_document = self.inventor_app.Documents.Add(
            win32.constants.kPartDocumentObject, "", True
        )
        self.part_doc = win32.CastTo(self.inventor_document, "PartDocument")
        self.comp_def = self.part_doc.ComponentDefinition

    def open_document(self, file_path):
        doc = self.inventor_app.Documents.Open(file_path)
        
        try:
            # Cast to proper PartDocument type using the COM wrapper
            self.part_doc = self.mod.PartDocument(doc)
            self.comp_def = self.part_doc.ComponentDefinition
            self.inventor_document = doc
            return self.part_doc
        except Exception:
            raise TypeError(
                f"File '{file_path}' is not a valid Inventor Part Document."
            )

    def work_plane(self, name: str = "XY", origin: Optional[VectorLike] = None, normal: Optional[VectorLike] = None, offset: Optional[float] = None, **kwargs) -> "InventorWorkPlane":
        """
        Create a workplane either by name or by origin and normal vector.
        
        Args:
            name: Standard plane name ("XY", "XZ", "YZ") - used if origin/normal not provided
            origin: Origin point for custom workplane
            normal: Normal vector for custom workplane
            offset: Offset distance for the workplane
            **kwargs: Additional arguments (like app parameter)
            
        Returns:
            InventorWorkPlane instance
        """
        from rapidcadpy.integrations.inventor.workplane import InventorWorkPlane
        
        if origin is not None and normal is not None:
            # Create custom workplane from origin and normal
            return InventorWorkPlane.from_origin_normal(origin, normal, app=self)
        elif offset is not None and name in ["XY", "XZ", "YZ"]:
            # Create standard named workplane with offset
            return InventorWorkPlane.create_offset_plane(app=self, name=name, offset=offset)
        elif name == "XY":
            # Create standard XY workplane at origin
            return InventorWorkPlane.xy_plane(app=self)
        else:
            # Create standard named workplane
            return super().work_plane(name)  # type: ignore
