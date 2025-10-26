import os
from typing import TYPE_CHECKING, Optional

from rapidcadpy.app import App
from rapidcadpy.cad_types import VectorLike

if TYPE_CHECKING:
    from rapidcadpy.integrations.inventor.workplane import InventorWorkPlane


class InventorApp(App):
    def __init__(self, headless: bool = False):
        try:
            import win32com.client as win32
            from win32com.client import Dispatch, gencache
        except ImportError:
            raise ImportError(
                "pywin32 is required for Inventor integration. Install with: pip install pywin32"
            )

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

    def work_plane(
        self,
        name: str = "XY",
        origin: Optional[VectorLike] = None,
        normal: Optional[VectorLike] = None,
        offset: Optional[float] = None,
        **kwargs,
    ) -> "InventorWorkPlane":
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
            return InventorWorkPlane.create_offset_plane(
                app=self, name=name, offset=offset
            )
        elif name == "XY":
            # Create standard XY workplane at origin
            return InventorWorkPlane.xy_plane(app=self)
        else:
            # Create standard named workplane
            return super().work_plane(name)  # type: ignore
        
    def to_stl(self, file_name: str) -> None:
        """
        Export the shape to STL format using Autodesk Inventor COM API.
        
        Uses the STL TranslatorAddIn with proper type casting to enable
        SaveCopyAs method with export options (resolution, format, units).

        Args:
            file_name: Path to the output STL file
        """
        try:
            import win32com.client as win32
            import pythoncom
        except ImportError:
            raise ImportError(
                "pywin32 is required for Inventor integration. Install with: pip install pywin32"
            )

        # Initialize COM library for this thread
        pythoncom.CoInitialize()
        
        try:
            # Ensure the file has .stl extension
            if not file_name.lower().endswith(".stl"):
                file_name += ".stl"

            # Convert to absolute path
            file_path = os.path.abspath(file_name)
            active_doc = self.inventor_app.ActiveDocument

            # Verify it's a part document
            if active_doc.DocumentType != win32.constants.kPartDocumentObject:
                raise RuntimeError("Active document must be a Part document for STL export")

            # Get the STL Translator add-in (GUID is fixed for STL translator)
            stl_addin = self.inventor_app.ApplicationAddIns.ItemById(
                "{533E9A98-FC3B-11D4-8E7E-0010B541CD80}"
            )
            
            if stl_addin is None:
                raise RuntimeError("STL translator add-in not found in Inventor")
            
            # Cast to TranslatorAddIn using the type library module
            # This is crucial - it makes methods like SaveCopyAs available
            stl_translator = self.mod.TranslatorAddIn(stl_addin)

            # Create translation context and options
            ctx = self.transient_obj.CreateTranslationContext()
            ctx.Type = win32.constants.kFileBrowseIOMechanism
            
            # Create options map to check if translator supports this document
            opts_check = self.transient_obj.CreateNameValueMap()

            # Check if translator has save options for this document
            if stl_translator.HasSaveCopyAsOptions(active_doc, ctx, opts_check):
                # Create a new NameValueMap for actual export options
                opts = self.transient_obj.CreateNameValueMap()
                opts.Add("Resolution", 2)  # 0=Low, 1=Medium, 2=High, 3=Custom
                opts.Add("BinaryFormat", True)  # Binary STL (more efficient)
                opts.Add("ExportUnits", 2)  # kMillimeterLengthUnits = 2
                
                # Create data medium for output file
                data = self.transient_obj.CreateDataMedium()
                data.FileName = file_path

                # Perform the export
                stl_translator.SaveCopyAs(active_doc, ctx, opts, data)
                print(f"Successfully exported STL to: {file_path}")
            else:
                raise RuntimeError("STL translator has no save options for this document")

            # Verify the file was created
            if not os.path.exists(file_path):
                raise RuntimeError(f"STL file was not created at: {file_path}")
                
        except Exception as e:
            # Re-raise with more context if it's not already a RuntimeError
            if not isinstance(e, RuntimeError):
                raise RuntimeError(f"Failed to export STL: {str(e)}")
            raise
        finally:
            # Uninitialize COM library
            pythoncom.CoUninitialize()

