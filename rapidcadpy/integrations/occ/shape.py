from rapidcadpy.shape import Shape


class OccShape(Shape):
    def __init__(self, obj, app) -> None:
        self.app = app
        super().__init__(obj, app)

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

    def to_step(self, file_name: str) -> None:
        raise NotImplementedError("STEP export not implemented yet.")

    def to_png(
        self, 
        file_name: str, 
        view: str = "iso",
        width: int = 800,
        height: int = 600,
        backend: str = "auto"
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
            backend: Rendering backend - "auto", "pyribbit", "vedo" (default: "auto")
                    Note: "trimesh" is supported but requires pyribbit

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
                backends = ["pyribbit", "vedo"]  # trimesh requires pyribbit anyway
            else:
                backends = [backend]
            
            rendered = False
            last_error = None
            
            for backend_name in backends:
                try:
                    if backend_name == "trimesh":
                        self._render_with_trimesh(tmp_stl, file_name, view, width, height)
                        rendered = True
                        break
                    elif backend_name == "pyribbit":
                        self._render_with_pyribbit(tmp_stl, file_name, view, width, height)
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
                    "pyribbit (pip install pyribbit), "
                    "vedo (pip install vedo). "
                    f"Last error: {last_error}"
                )
                
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_stl):
                os.remove(tmp_stl)

    
    def _render_with_pyribbit(self, stl_file: str, output: str, view: str, width: int, height: int):
        """Render using pyribbit (high quality, physically based - updated fork of pyrender)."""
        import trimesh
        import pyribbit
        import numpy as np
        from PIL import Image
        
        # Load mesh
        mesh = trimesh.load(stl_file)
        mesh = pyribbit.Mesh.from_trimesh(mesh)
        
        # Create scene
        scene = pyribbit.Scene(ambient_light=[0.5, 0.5, 0.5])  # Increased ambient light
        scene.add(mesh)
        
        # Set camera
        camera = pyribbit.PerspectiveCamera(yfov=np.pi / 3.0)
        
        # Camera position based on view
        bounds = mesh.bounds
        center = bounds.mean(axis=0)
        scale = np.max(bounds[1] - bounds[0]) * 2.5  # Distance from object
        
        # Define camera position and up vector for each view
        if view == "iso":
            # Isometric view from top-right-front
            cam_pos = center + np.array([scale, scale, scale]) * 0.7
            up = np.array([0, 0, 1])
        elif view == "front":
            # Looking along +Y axis (from front)
            cam_pos = center + np.array([0, -scale, 0])
            up = np.array([0, 0, 1])
        elif view == "top":
            # Looking down -Z axis (from top)
            cam_pos = center + np.array([0, 0, scale])
            up = np.array([0, 1, 0])
        elif view == "right":
            # Looking along -X axis (from right)
            cam_pos = center + np.array([scale, 0, 0])
            up = np.array([0, 0, 1])
        
        # Create lookAt matrix
        # Camera looks at the center of the object
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up_corrected = np.cross(right, forward)
        
        camera_pose = np.eye(4)
        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = up_corrected
        camera_pose[:3, 2] = -forward
        camera_pose[:3, 3] = cam_pos
        
        # Add directional light from camera direction
        light = pyribbit.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        light_pose = camera_pose.copy()
        scene.add(light, pose=light_pose)
        
        scene.add(camera, pose=camera_pose)
        
        # Render
        renderer = pyribbit.OffscreenRenderer(width, height)
        color, _ = renderer.render(scene)
        
        # Save
        Image.fromarray(color).save(output)
        renderer.delete()
    
    def _render_with_vedo(self, stl_file: str, output: str, view: str, width: int, height: int):
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
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

        cut_result = BRepAlgoAPI_Cut(self.obj, other.obj)
        cut_result.Build()
        if not cut_result.IsDone():
            raise RuntimeError("Cut operation failed.")

        # Update the current object with the cut result (in-place modification)
        self.obj = cut_result.Shape()
        return self

    def union(self, other: Shape) -> Shape:
        """
        Perform a boolean union operation (addition) on this shape.

        This operation modifies the current shape in-place by unioning
        it with the other shape.

        Args:
            other: The shape to union with this shape

        Returns:
            OccShape: Self (modified in-place) for method chaining
        """
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse

        fuse_result = BRepAlgoAPI_Fuse(self.obj, other.obj)
        fuse_result.Build()
        if not fuse_result.IsDone():
            raise RuntimeError("Union operation failed.")

        # Update the current object with the union result (in-place modification)
        self.obj = fuse_result.Shape()
        return self
