from rapidcadpy.app import App


class InventorApp(App):
    def __init__(self):
        import win32com.client as win32
        from win32com.client import Dispatch, gencache

        from rapidcadpy.integrations.inventor.workplane import InventorWorkPlane

        super().__init__(InventorWorkPlane)
        try:
            self.inventor_app = win32.GetActiveObject("Inventor.Application")
        except Exception:
            self.inventor_app = Dispatch("Inventor.Application")
            self.inventor_app.Visible = True

        self.mod = gencache.EnsureModule(
            "{D98A091D-3A0F-4C3E-B36E-61F62068D488}", 0, 1, 0
        )

        self.transient_geom = self.inventor_app.TransientGeometry
        self.transient_obj = self.inventor_app.TransientObjects

        # Do NOT create a new document here!
        self.comp_def = None
        self.part_doc = None

    def new_document(self):
        import win32com.client as win32

        self.inventor_document = self.inventor_app.Documents.Add(
            win32.constants.kPartDocumentObject, "", True
        )
        self.part_doc = self.mod.PartDocument(self.inventor_document)
        self.comp_def = self.part_doc.ComponentDefinition

    def open_document(self, file_path):
        doc = self.inventor_app.Documents.Open(file_path)
        part_doc = self.mod.PartDocument(doc)
        # Use COM type check to ensure it's a PartDocument
        if not hasattr(part_doc, "ComponentDefinition"):
            raise TypeError(
                f"File '{file_path}' is not a valid Inventor Part Document."
            )
        return part_doc

    def work_plane(self, name: str = "XY") -> "InventorWorkPlane":
        return super().work_plane(name)  # type: ignore
