from typing import List, Union, Optional
from rapidcadpy.shape import Shape


class OccShape(Shape):
    def __init__(self, obj, app, material=None) -> None:
        self.app = app
        super().__init__(obj, app, material)
        # Register this shape with the app

    def volume(self) -> float:
        """
        Calculate the volume of the shape using OpenCASCADE's GProp_GProps.
        
        Returns:
            float: Volume in cubic units
            
        Raises:
            RuntimeError: If volume calculation fails
        """
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(self.obj, props)
        return props.Mass()

    def to_stl(self, file_name: str):
        # The constructor used here automatically calls mesh.Perform(). https://dev.opencascade.org/doc/refman/html/class_b_rep_mesh___incremental_mesh.html#a3a383b3afe164161a3aa59a492180ac6
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.StlAPI import StlAPI_Writer

        tolerance = 1e-3
        angular_tolerance = 0.1
        ascii = False
        relative = True
        parallel = True
        BRepMesh_IncrementalMesh(
            self.obj, tolerance, relative, angular_tolerance, parallel
        )
        writer = StlAPI_Writer()
        writer.ASCIIMode = ascii

        return writer.Write(self.obj, file_name)

    def to_step(self, file_name: str, heal: bool = False) -> None:
        """
        Export shape to STEP file.
        
        Args:
            file_name: Path to save the STEP file
            heal: If True, heal/fix geometry before export (useful for FEA meshing)
        """
        from OCP.STEPControl import STEPControl_Writer, STEPControl_StepModelType
        from OCP.IFSelect import IFSelect_RetDone

        shape_to_export = self.obj
        
        if heal and False:
            # Light healing - just fix shape issues without destroying topology
            from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Solid
            from OCP.TopoDS import TopoDS
            from OCP.TopAbs import TopAbs_SOLID
            from OCP.TopExp import TopExp_Explorer
            
            # Fix solid issues
            fixer = ShapeFix_Shape(shape_to_export)
            fixer.SetPrecision(1e-6)
            fixer.Perform()
            shape_to_export = fixer.Shape()
            
            # Fix each solid individually
            explorer = TopExp_Explorer(shape_to_export, TopAbs_SOLID)
            while explorer.More():
                solid = TopoDS.Solid_s(explorer.Current())
                solid_fixer = ShapeFix_Solid(solid)
                solid_fixer.Perform()
                explorer.Next()

        step_writer = STEPControl_Writer()
        step_writer.Transfer(shape_to_export, STEPControl_StepModelType.STEPControl_AsIs)
        status = step_writer.Write(file_name)
        if status != IFSelect_RetDone:
            raise RuntimeError("Failed to write STEP file.")

    def to_png(
        self,
        file_name: Optional[str] = None,
        view: str = "iso",
        width: int = 800,
        height: int = 600,
        backend: str = "auto",
    ) -> None:
        """
        Render the shape to a PNG file with 3D shading, or display in interactive window.

        This method exports the shape to STL, then renders it with proper lighting
        and shading. Works in headless environments (no display required) when file_name
        is provided. If file_name is None, opens an interactive 3D viewer window.

        Args:
            file_name: Path to save the PNG file. If None, displays in interactive window.
            view: Camera view - "iso", "front", "top", "right", or "X", "Y", "Z"
            width: Image width in pixels (default: 800)
            height: Image height in pixels (default: 600)
            backend: Rendering backend - "auto", "pyvista", "vedo" (default: "auto")

        Raises:
            ImportError: If no supported rendering library is available
            ValueError: If view is invalid
        """
        import os
        import tempfile

        # Map view names
        view_map = {
            "iso": "iso",
            "isometric": "iso",
            "front": "front",
            "y": "front",
            "top": "top",
            "z": "top",
            "right": "right",
            "x": "right",
        }

        view = view.lower()
        if view not in view_map:
            raise ValueError(
                f"View must be one of {list(view_map.keys())}, got '{view}'"
            )
        view = view_map[view]

        # Create temporary STL file
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            tmp_stl = tmp.name

        try:
            # Export to STL
            self.to_stl(tmp_stl)

            # Try rendering backends in order
            if backend == "auto":
                backends = ["pyvista", "vedo"]
            else:
                backends = [backend]

            rendered = False
            last_error = None

            for backend_name in backends:
                try:
                    if backend_name == "pyvista":
                        self._render_with_pyvista(
                            tmp_stl, file_name, view, width, height
                        )
                        rendered = True
                        break
                    elif backend_name == "vedo":
                        self._render_with_vedo(tmp_stl, file_name, view, width, height)
                        rendered = True
                        break
                except ImportError as e:
                    last_error = e
                    continue

            if not rendered:
                raise ImportError(
                    "No rendering backend available. Install one of: "
                    "pyvista (pip install pyvista), "
                    "vedo (pip install vedo). "
                    f"Last error: {last_error}"
                )

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_stl):
                os.remove(tmp_stl)

    def _render_with_pyvista(
        self, stl_file: str, output: Optional[str], view: str, width: int, height: int
    ):
        """Render using PyVista (fast, simple, and lightweight)."""
        import pyvista as pv

        # Enable off-screen rendering only if saving to file
        if output:
            pv.OFF_SCREEN = True

        # Load mesh
        mesh = pv.read(stl_file)

        # Create plotter
        plotter = pv.Plotter(off_screen=(output is not None), window_size=[width, height])
        plotter.add_mesh(mesh, color="lightgray", show_edges=False)
        plotter.background_color = "white"

        # Set camera based on view
        if view == "iso":
            # Isometric view
            plotter.camera_position = "iso"
        elif view == "front":
            # Front view (looking along +Y)
            plotter.camera.position = (0, -10, 0)
            plotter.camera.focal_point = (0, 0, 0)
            plotter.camera.up = (0, 0, 1)
        elif view == "top":
            # Top view (looking down -Z)
            plotter.camera.position = (0, 0, 10)
            plotter.camera.focal_point = (0, 0, 0)
            plotter.camera.up = (0, 1, 0)
        elif view == "right":
            # Right view (looking along +X)
            plotter.camera.position = (10, 0, 0)
            plotter.camera.focal_point = (0, 0, 0)
            plotter.camera.up = (0, 0, 1)

        # Render and save or show
        if output:
            plotter.show(screenshot=output, auto_close=True)
        else:
            plotter.show()

    def _render_with_vedo(
        self, stl_file: str, output: Optional[str], view: str, width: int, height: int
    ):
        """Render using vedo (feature-rich, scientific visualization)."""
        import vedo

        # Load mesh
        mesh = vedo.load(stl_file)

        if output:
            # Create plotter (offscreen)
            plotter = vedo.Plotter(offscreen=True, size=(width, height))
            plotter.show(mesh, viewup="z")

            # Set camera view
            if view == "iso":
                plotter.camera.Azimuth(45)
                plotter.camera.Elevation(30)
            elif view == "front":
                plotter.camera.Azimuth(0)
                plotter.camera.Elevation(0)
            elif view == "top":
                plotter.camera.Azimuth(0)
                plotter.camera.Elevation(90)
            elif view == "right":
                plotter.camera.Azimuth(90)
                plotter.camera.Elevation(0)

            # Render and save
            plotter.screenshot(output)
            plotter.close()
        else:
            # Interactive mode
            plotter = vedo.Plotter(size=(width, height))
            
            # Set camera view
            if view == "iso":
                plotter.show(mesh, viewup="z", camera={'azimuth': 45, 'elevation': 30})
            elif view == "front":
                plotter.show(mesh, viewup="z", camera={'azimuth': 0, 'elevation': 0})
            elif view == "top":
                plotter.show(mesh, viewup="z", camera={'azimuth': 0, 'elevation': 90})
            elif view == "right":
                plotter.show(mesh, viewup="z", camera={'azimuth': 90, 'elevation': 0})
            else:
                plotter.show(mesh, viewup="z")

    def cut(self, other: "OccShape") -> "OccShape":
        """
        Perform a boolean cut operation (subtraction) on this shape.

        This operation modifies the current shape in-place by subtracting
        the other shape from it.

        Args:
            other: The shape to subtract from this shape

        Returns:
            OccShape: Self (modified in-place) for method chaining
        """
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

        cut_result = BRepAlgoAPI_Cut(self.obj, other.obj)
        cut_result.Build()
        if not cut_result.IsDone():
            raise RuntimeError("Cut operation failed.")

        # Update the current object with the cut result (in-place modification)
        self.obj = cut_result.Shape()
        return self

    def union(self, other: Union[Shape, List[Shape]]) -> Shape:
        """
        Perform a boolean union operation (addition) with one or more shapes.

        This operation modifies the current shape in-place by unioning
        it with the other shape(s).

        Args:
            other: A single Shape or a list of Shapes to union with this shape

        Returns:
            OccShape: Self (modified in-place) for method chaining
            
        Examples:
            # Union with a single shape
            result = shape1.union(shape2)
            
            # Union with multiple shapes
            result = shape1.union([shape2, shape3, shape4])
        """
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse

        # Convert single shape to list for uniform handling
        shapes_to_union = [other] if not isinstance(other, list) else other
        
        # Perform union with each shape sequentially
        for shape in shapes_to_union:
            fuse_result = BRepAlgoAPI_Fuse(self.obj, shape.obj)
            fuse_result.Build()
            if not fuse_result.IsDone():
                raise RuntimeError("Union operation failed.")
            
            # Update the current object with the union result (in-place modification)
            self.obj = fuse_result.Shape()
        
        return self

    def get_fea_analyzer(self, material, mesh_size=2.0, element_type='tet4'):
        """
        Get torch-fem analyzer for this OccShape.
        
        Args:
            material: Material properties
            mesh_size: Target mesh element size
            element_type: Element type
        
        Returns:
            OccTorchFEMAnalyzer instance, or None if dependencies unavailable
        """
        try:
            from rapidcadpy.fea.fea_analyzer import OccTorchFEMAnalyzer
            
            if OccTorchFEMAnalyzer.is_available():
                return OccTorchFEMAnalyzer(self, material, mesh_size, element_type)
            else:
                import warnings
                warnings.warn(
                    "torch-fem dependencies not available. "
                    "Install with: pip install rapidcadpy[fea]"
                )
                return None
        except ImportError:
            return None
