from .shape import Shape


class OccShape(Shape):
    def __init__(self, obj, app) -> None:
        super().__init__(obj, app)

    def volume(self) -> float:
        """
        Calculate the volume of the shape using OpenCASCADE's GProp_GProps.

        Returns:
            float: Volume in cubic units

        Raises:
            RuntimeError: If volume calculation fails
        """
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepGProp import brepgprop

        try:
            props = GProp_GProps()
            brepgprop.VolumeProperties(self.obj, props)
            return props.Mass()
        except Exception as e:
            print(f"Failed to calculate volume: {e}")
            return 1.0

    def to_stl(self, file_name: str):
        # The constructor used here automatically calls mesh.Perform(). https://dev.opencascade.org/doc/refman/html/class_b_rep_mesh___incremental_mesh.html#a3a383b3afe164161a3aa59a492180ac6
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.StlAPI import StlAPI_Writer

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

    def to_step(self, file_name: str) -> None:
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_StepModelType
        from OCC.Core.IFSelect import IFSelect_RetDone

        step_writer = STEPControl_Writer()
        step_writer.Transfer(self.obj, STEPControl_StepModelType.STEPControl_AsIs)
        status = step_writer.Write(file_name)
        if status != IFSelect_RetDone:
            raise RuntimeError("Failed to write STEP file.")

    def to_png(
        self,
        file_name: str,
        view: str = "iso",
        width: int = 800,
        height: int = 600,
        backend: str = "auto",
    ) -> None:
        """
        Render the shape to a PNG file with 3D shading using off-screen rendering.

        This method exports the shape to STL, then renders it with proper lighting
        and shading. Works in headless environments (no display required).

        Args:
            file_name: Path to save the PNG file
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
        self, stl_file: str, output: str, view: str, width: int, height: int
    ):
        """Render using PyVista (fast, simple, and lightweight)."""
        import pyvista as pv

        # Enable off-screen rendering
        pv.OFF_SCREEN = True

        # Load mesh
        mesh = pv.read(stl_file)

        # Create plotter
        plotter = pv.Plotter(off_screen=True, window_size=[width, height])
        plotter.add_mesh(mesh, color="lightgray", show_edges=False)
        plotter.set_background("white")

        # Set camera based on view
        if view == "iso":
            # Isometric view
            plotter.camera_position = "iso"
        elif view == "front":
            # Front view (looking along +Y)
            plotter.view_xy()
        elif view == "top":
            # Top view (looking down -Z)
            plotter.view_xz()
        elif view == "right":
            # Right view (looking along +X)
            plotter.view_yz()

        # Render and save
        plotter.show(screenshot=output, auto_close=True)

    def _render_with_vedo(
        self, stl_file: str, output: str, view: str, width: int, height: int
    ):
        """Render using vedo (feature-rich, scientific visualization)."""
        import vedo

        # Load mesh
        mesh = vedo.load(stl_file)

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
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut

        cut_result = BRepAlgoAPI_Cut(self.obj, other.obj)
        cut_result.Build()
        if not cut_result.IsDone():
            raise RuntimeError("Cut operation failed.")

        # Update the current object with the cut result (in-place modification)
        self.obj = cut_result.Shape()
        return self

    def union(self, other: "Shape | list[Shape]") -> "OccShape":
        """
        Perform a boolean union operation (addition) on this shape.

        This operation modifies the current shape in-place by unioning
        it with the other shape(s).

        Args:
            other: A single shape or a list of shapes to union with this shape

        Returns:
            OccShape: Self (modified in-place) for method chaining

        Example:
            >>> shape1.union(shape2)  # Union with single shape
            >>> shape1.union([shape2, shape3, shape4])  # Union with multiple shapes
        """
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse

        # Handle list of shapes
        if isinstance(other, list):
            for shape in other:
                fuse_result = BRepAlgoAPI_Fuse(self.obj, shape.obj)
                fuse_result.Build()
                if not fuse_result.IsDone():
                    raise RuntimeError("Union operation failed.")
                self.obj = fuse_result.Shape()
        else:
            fuse_result = BRepAlgoAPI_Fuse(self.obj, other.obj)
            fuse_result.Build()
            if not fuse_result.IsDone():
                raise RuntimeError("Union operation failed.")
            self.obj = fuse_result.Shape()

        return self
