import os

from win32com.client import constants

from rapidcadpy.shape import Shape


class InventorShape(Shape):
    def __init__(self, obj, app) -> None:
        super().__init__(obj, app)
        self.app: "InventorApp" = app  # type: ignore

    def volume(self) -> float:
        return 1.0  # Placeholder implementation

    def to_stl(self, file_name: str) -> None:
        """
        Export the shape to STL format using Autodesk Inventor COM API.

        Args:
            file_name: Path to the output STL file
        """
        try:
            import win32com.client as win32
        except ImportError:
            raise ImportError(
                "pywin32 is required for Inventor integration. Install with: pip install pywin32"
            )

        # Ensure the file has .stl extension
        if not file_name.lower().endswith(".stl"):
            file_name += ".stl"

        # Convert to absolute path
        file_path = os.path.abspath(file_name)
        active_doc = self.app.inventor_app.ActiveDocument

        # Verify it's a part document
        if active_doc.DocumentType != win32.constants.kPartDocumentObject:
            raise RuntimeError("Active document must be a Part document for STL export")

        # Create a TranslatorAddIn object for STL export
        translator_addins = self.app.inventor_app.ApplicationAddIns.ItemById(
            "{533E9A98-FC3B-11D4-8E7E-0010B541CD80}"
        )

        if translator_addins is None:
            raise RuntimeError("STL translator add-in not found in Inventor")

        # Create STL translator
        stl_translator = translator_addins.Object

        # Create a translation context
        translation_context = self.app.transient_obj.CreateTranslationContext()

        # Set the context type to STL
        translation_context.Type = win32.constants.kFileTranslationContext

        # Create options object for STL export
        options = self.app.transient_obj.CreateNameValueMap()

        # Set STL export options
        options.set_Value("Resolution", 0)  # 0 = High, 1 = Medium, 2 = Low
        options.set_Value("SurfaceDeviation", 0.0001)  # Surface deviation tolerance
        options.set_Value("NormalDeviation", 15.0)  # Normal deviation in degrees
        options.set_Value("MaxEdgeLength", 0.0)  # Maximum edge length (0 = no limit)
        options.set_Value("AspectRatio", 0.0)  # Aspect ratio (0 = no limit)
        options.set_Value("ExportUnits", 6)  # Units: 6 = millimeters
        options.set_Value("ExportFileStructure", 0)  # 0 = Single file

        # Set the options in the context
        if hasattr(stl_translator, "SetOptions"):
            stl_translator.SetOptions(translation_context, options)

        # Create a DataMedium object for the output file
        data_medium = self.app.transient_obj.CreateDataMedium()
        data_medium.FileName = file_path

        # Perform the translation
        try:
            stl_translator.SaveCopyAs(
                active_doc, translation_context, options, data_medium
            )
            print(f"Successfully exported STL to: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to export STL: {str(e)}")

        # Verify the file was created
        if not os.path.exists(file_path):
            raise RuntimeError(f"STL file was not created at: {file_path}")

    def to_step(self, file_name: str) -> None:
        """
        Export the shape to STEP format using Autodesk Inventor COM API.

        Args:
            file_name: Path to the output STEP file
        """
        try:
            import win32com.client as win32
        except ImportError:
            raise ImportError(
                "pywin32 is required for Inventor integration. Install with: pip install pywin32"
            )

        # Ensure the file has .step or .stp extension
        if not (
            file_name.lower().endswith(".step") or file_name.lower().endswith(".stp")
        ):
            file_name += ".step"

        # Convert to absolute path
        file_path = os.path.abspath(file_name)
        active_doc = self.app.inventor_app.ActiveDocument

        # Verify it's a part document
        if active_doc.DocumentType != win32.constants.kPartDocumentObject:
            raise RuntimeError(
                "Active document must be a Part document for STEP export"
            )

        # Create a TranslatorAddIn object for STEP export
        # STEP translator GUID
        translator_addins = self.app.inventor_app.ApplicationAddIns.ItemById(
            "{90AF7F40-0C01-11D5-8E83-0010B541CD80}"
        )

        if translator_addins is None:
            raise RuntimeError("STEP translator add-in not found in Inventor")

        # Create STEP translator
        step_translator = translator_addins.Object

        # Create a translation context
        translation_context = self.app.transient_obj.CreateTranslationContext()

        # Set the context type to file translation
        translation_context.Type = win32.constants.kFileTranslationContext

        # Create options object for STEP export
        options = self.app.transient_obj.CreateNameValueMap()

        # Set STEP export options
        options.set_Value("ApplicationProtocolType", 1)  # 1 = AP214, 2 = AP203
        options.set_Value("Description", "Exported from RapidCAD-Py")
        options.set_Value("Author", "RapidCAD-Py")
        options.set_Value("Organization", "")
        options.set_Value("ExportUnits", 6)  # Units: 6 = millimeters

        # Set the options in the context
        if hasattr(step_translator, "SetOptions"):
            step_translator.SetOptions(translation_context, options)

        # Create a DataMedium object for the output file
        data_medium = self.app.transient_obj.CreateDataMedium()
        data_medium.FileName = file_path

        # Perform the translation
        try:
            step_translator.SaveCopyAs(
                active_doc, translation_context, options, data_medium
            )
            print(f"Successfully exported STEP to: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to export STEP: {str(e)}")

        # Verify the file was created
        if not os.path.exists(file_path):
            raise RuntimeError(f"STEP file was not created at: {file_path}")

    def to_ipt(self, file_name: str) -> None:
        """
        Save the current document as an Inventor Part (.ipt) file using Autodesk Inventor COM API.

        Args:
            file_name: Path to the output IPT file
        """
        try:
            import win32com.client as win32
        except ImportError:
            raise ImportError(
                "pywin32 is required for Inventor integration. Install with: pip install pywin32"
            )

        # Ensure the file has .ipt extension
        if not file_name.lower().endswith(".ipt"):
            file_name += ".ipt"

        # Convert to absolute path
        file_path = os.path.abspath(file_name)
        active_doc = self.app.inventor_app.ActiveDocument

        # Verify it's a part document
        if active_doc.DocumentType != win32.constants.kPartDocumentObject:
            raise RuntimeError("Active document must be a Part document for IPT save")

        try:
            # Use SaveAs method to save the document as IPT
            # This saves the current document state to a new IPT file
            active_doc.SaveAs(file_path, False)  # False = don't save as copy
            print(f"Successfully saved IPT to: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save IPT: {str(e)}")

        # Verify the file was created
        if not os.path.exists(file_path):
            raise RuntimeError(f"IPT file was not created at: {file_path}")

    def cut(self, other: "Shape") -> "Shape":
        """
        Perform a boolean cut operation between this shape and another shape.

        Args:
            other: The shape to subtract from this shape

        Returns:
            InventorShape: The resulting shape after the cut operation
        """
        comp_def = self.app.comp_def

        # Get the surface bodies from the component definition
        target_body = self.obj
        tool_body = other.obj

        # Debug: Print information about the bodies
        print(f"Target body: {target_body}, Type: {type(target_body)}")
        print(f"Tool body: {tool_body}, Type: {type(tool_body)}")
        print(f"Target solid: {target_body.IsSolid}")
        print(f"Tool solid: {tool_body.IsSolid}")
        print(f"Surface bodies count: {comp_def.SurfaceBodies.Count}")

        # Ensure we're working with surface bodies
        if not hasattr(target_body, "IsSolid") or not hasattr(tool_body, "IsSolid"):
            raise ValueError(
                "Both objects must be surface bodies for boolean operations."
            )

        if not target_body.IsSolid or not tool_body.IsSolid:
            raise ValueError("Both bodies must be solid for boolean operations.")

        # Check if bodies are in the component definition
        target_found = False
        tool_found = False
        for i in range(1, comp_def.SurfaceBodies.Count + 1):
            body = comp_def.SurfaceBodies.Item(i)
            if body == target_body:
                target_found = True
            if body == tool_body:
                tool_found = True

        if not target_found:
            raise ValueError("Target body not found in component definition")
        if not tool_found:
            raise ValueError("Tool body not found in component definition")

        # Create object collection for the tool bodies
        oc = self.app.transient_obj.CreateObjectCollection()
        oc.Add(tool_body)

        try:
            # Perform the cut operation
            feat = comp_def.Features.CombineFeatures.Add(
                target_body,  # BaseBody (SurfaceBody)
                oc,  # ToolBodies (ObjectCollection)
                constants.kCutOperation,  # Operation type
                # Note: Not passing KeepToolBodies parameter - let it default
            )

            # Return a new instance with the modified target body
            return self.__class__(target_body, self.app)

        except Exception as e:
            # More detailed error information
            error_msg = "Boolean cut operation failed. "
            error_msg += f"Target body: {target_body}, "
            error_msg += f"Tool body: {tool_body}, "
            error_msg += f"Bodies in comp def: {comp_def.SurfaceBodies.Count}, "
            error_msg += f"Error: {str(e)}"
            raise RuntimeError(error_msg)

    def union(self, other: "Shape") -> "Shape":
        """
        Perform a boolean union operation between this shape and another shape.

        Args:
            other: The shape to union with this shape

        Returns:
            InventorShape: The resulting shape after the union operation
        """
        comp_def = self.app.comp_def

        # Get the surface bodies - they should be SurfaceBody objects
        target_body = self.obj
        tool_body = other.obj

        # Ensure we're working with surface bodies
        if not hasattr(target_body, "IsSolid") or not hasattr(tool_body, "IsSolid"):
            raise ValueError(
                "Both objects must be surface bodies for boolean operations."
            )

        if not target_body.IsSolid or not tool_body.IsSolid:
            raise ValueError("Both bodies must be solid for boolean operations.")

        # Create object collection for the tool bodies
        oc = self.app.transient_obj.CreateObjectCollection()
        oc.Add(tool_body)

        try:
            # Perform the union operation - use the correct parameter order
            # Add(BaseBody, ToolBodies, Operation, KeepToolBodies)
            feat = comp_def.Features.CombineFeatures.Add(
                target_body,  # BaseBody (SurfaceBody)
                oc,  # ToolBodies (ObjectCollection)
                constants.kJoinOperation,  # Operation type
                False,  # KeepToolBodies (don't keep tool bodies)
            )

            # Return a new instance with the modified target body
            return self.__class__(target_body, self.app)

        except Exception as e:
            # More detailed error information
            error_msg = "Boolean union operation failed. "
            error_msg += f"Target body type: {type(target_body).__name__}, "
            error_msg += f"Tool body type: {type(tool_body).__name__}, "
            error_msg += f"Target solid: {getattr(target_body, 'IsSolid', 'Unknown')}, "
            error_msg += f"Tool solid: {getattr(tool_body, 'IsSolid', 'Unknown')}, "
            error_msg += f"Error: {str(e)}"
            raise RuntimeError(error_msg)

    def to_png(self, file_name: str, view: str = "iso", width: int = 800, height: int = 600, backend: str = "auto") -> None:
        raise NotImplementedError("to_png is not implemented for InventorShape")