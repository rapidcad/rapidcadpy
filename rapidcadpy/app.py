import tempfile
from typing import TYPE_CHECKING, List, Optional, Type, Union

from rapidcadpy.fea.results import FEAResults

if TYPE_CHECKING:
    pass

from rapidcadpy.fea.boundary_conditions import BoundaryCondition, Load
from rapidcadpy.fea.materials import MaterialProperties
from rapidcadpy.shape import Shape
from rapidcadpy.workplane import Workplane


class App:
    def __init__(self, work_plane_class: Type[Workplane] = Workplane):
        self.work_plane_class = work_plane_class
        self._workplanes: List[Workplane] = []
        self._shapes: List["Shape"] = []

    def work_plane(self, name: str, offset: Optional[float] = None) -> Workplane:
        if name.upper() == "XY":
            return self.work_plane_class.xy_plane(app=self, offset=offset)
        elif name.upper() == "XZ":
            return self.work_plane_class.xz_plane(app=self, offset=offset)
        elif name.upper() == "YZ":
            return self.work_plane_class.yz_plane(app=self, offset=offset)
        else:
            raise ValueError(f"Unknown workplane: {name}")

    def register_workplane(self, workplane: Workplane) -> None:
        """Register a workplane with this app for tracking."""
        if workplane not in self._workplanes:
            self._workplanes.append(workplane)

    def register_shape(self, shape: "Shape") -> None:
        """Register a shape with this app for tracking."""
        if shape not in self._shapes:
            self._shapes.append(shape)

    def get_workplanes(self) -> List[Workplane]:
        """Get all workplanes registered with this app."""
        return self._workplanes.copy()

    def get_shapes(self) -> List["Shape"]:
        """Get all shapes registered with this app."""
        return self._shapes.copy()

    def volume(self) -> float:
        """Get the total volume of all shapes registered with this app."""
        total_volume = 0.0
        for shape in self._shapes:
            if hasattr(shape, "volume"):
                total_volume += shape.volume()
        return total_volume

    def workplane_count(self) -> int:
        """Get the number of workplanes registered with this app."""
        return len(self._workplanes)

    def shape_count(self) -> int:
        """Get the number of shapes registered with this app."""
        return len(self._shapes)

    def new_document(self): ...

    def show_3d(
        self,
        width: int = 1200,
        height: int = 800,
        show_axes: bool = True,
        shape_opacity: float = 0.8,
        sketch_color: str = "red",
        screenshot: Optional[str] = None,
        show_edges: bool = False,
        camera_angle: str = "iso",
    ) -> None:
        """
        Visualize all shapes and sketches in 3D space using PyVista.

        Args:
            width: Window width in pixels (default: 1200)
            height: Window height in pixels (default: 800)
            show_axes: Whether to show coordinate axes (default: True)
            shape_opacity: Opacity of 3D shapes (0-1, default: 0.8)
            sketch_color: Color for 2D sketches (default: "red")
            screenshot: Optional path to save screenshot instead of showing interactively
            show_edges: Whether to show mesh edges/tessellation lines (default: False)
            camera_angle: Camera view angle - "iso" for isometric, "x", "y", or "z" for orthogonal plane views (default: "iso")

        Raises:
            ImportError: If PyVista is not installed
            ValueError: If no shapes or workplanes to display
        """
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError(
                "PyVista is required for 3D visualization. Install with: pip install pyvista"
            )

        import os

        shapes = self.get_shapes()
        workplanes = self.get_workplanes()

        if not shapes and not workplanes:
            raise ValueError("No shapes or workplanes to display")

        # Create plotter - use off-screen mode if saving screenshot
        plotter = pv.Plotter(
            window_size=[width, height], off_screen=(screenshot is not None)
        )
        plotter.set_background("white")

        # Define a color palette for shapes
        shape_colors = [
            "lightblue",
            "lightgreen",
            "lightyellow",
            "lightcoral",
            "lightpink",
            "lightgray",
            "lavender",
            "peachpuff",
        ]

        # Add shapes to the plotter
        for idx, shape in enumerate(shapes):
            color = shape_colors[idx % len(shape_colors)]

            # If this is a wire/edge-only shape, STL export will be invalid and PyVista
            # will fail with "Empty meshes cannot be plotted". Render as a polyline.
            obj_type = type(getattr(shape, "obj", None)).__name__
            if obj_type in {"TopoDS_Wire", "TopoDS_Edge"}:
                try:
                    import numpy as np
                    from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve
                    from OCC.Core.GCPnts import GCPnts_UniformAbscissa

                    curve = BRepAdaptor_CompCurve(shape.obj)
                    absc = GCPnts_UniformAbscissa(curve, 2.0)  # ~2 unit spacing
                    if not absc.IsDone() or absc.NbPoints() < 2:
                        raise ValueError("Wire sampling produced too few points")

                    pts = []
                    for i in range(1, absc.NbPoints() + 1):
                        u = absc.Parameter(i)
                        p = curve.Value(u)
                        pts.append([p.X(), p.Y(), p.Z()])

                    pts_arr = np.array(pts)
                    polyline = pv.lines_from_points(pts_arr)
                    plotter.add_mesh(
                        polyline,
                        color=color,
                        line_width=5,
                        label=f"Shape {idx + 1}",
                        render_lines_as_tubes=True,
                    )
                    continue
                except Exception:
                    # Fall back to STL if the object isn't actually a valid wire.
                    pass

            # Export shape to temporary STL file
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                tmp_stl = tmp.name

            try:
                shape.to_stl(tmp_stl)
                mesh = pv.read(tmp_stl)
                plotter.add_mesh(
                    mesh,
                    color=color,
                    opacity=shape_opacity,
                    show_edges=show_edges,
                    edge_color="black" if show_edges else None,
                    label=f"Shape {idx + 1}",
                )
            finally:
                if os.path.exists(tmp_stl):
                    os.remove(tmp_stl)

        # Add sketches/workplanes to the plotter
        sketch_counter = 1
        for idx, workplane in enumerate(workplanes):
            # Add pending (not yet extruded) sketches
            if hasattr(workplane, "_pending_shapes") and workplane._pending_shapes:
                self._add_sketch_to_plotter(
                    plotter,
                    workplane,
                    workplane._pending_shapes,
                    sketch_color,
                    f"Sketch {sketch_counter}",
                )
                sketch_counter += 1

            # Add extruded sketches (sketches that were extruded but we want to show)
            if hasattr(workplane, "_extruded_sketches"):
                for extruded_sketch in workplane._extruded_sketches:
                    self._add_sketch_to_plotter(
                        plotter,
                        workplane,
                        extruded_sketch,
                        sketch_color,
                        f"Sketch {sketch_counter}",
                    )
                    sketch_counter += 1

        # Configure view
        if show_axes:
            plotter.add_axes()
        try:
            plotter.add_legend()
        except Exception as e:
            print(f"Error adding legend: {e}")

        # Set camera position based on camera_angle parameter
        if camera_angle.lower() == "iso":
            # Isometric view - view from corner
            plotter.camera_position = "iso"
        elif camera_angle.lower() == "x":
            # Looking along X axis (YZ plane view)
            plotter.view_yz()
        elif camera_angle.lower() == "y":
            # Looking along Y axis (XZ plane view)
            plotter.view_xz()
        elif camera_angle.lower() == "z":
            # Looking along Z axis (XY plane view - top down)
            plotter.view_xy()
        else:
            # Default to isometric if invalid angle provided
            plotter.camera_position = "iso"

        # Show or save
        if screenshot:
            # Off-screen rendering - save screenshot without opening window
            plotter.show(screenshot=screenshot)
        else:
            # Interactive mode - open window
            plotter.show()

    def _add_sketch_to_plotter(
        self, plotter, workplane, sketch_primitives, color: str, label: str
    ) -> None:
        """
        Add a 2D sketch to the PyVista plotter.

        Args:
            plotter: PyVista plotter instance
            workplane: The workplane the sketch belongs to (for coordinate transformation)
            sketch_primitives: List of 2D primitives (Line, Circle, Arc) from the sketch
            color: Color for the sketch
            label: Label for the sketch
        """
        import pyvista as pv
        import numpy as np
        from rapidcadpy.primitives import Line, Circle, Arc

        # Process each primitive in the sketch
        for primitive in sketch_primitives:
            if isinstance(primitive, Line):
                # Convert 2D line to 3D points using workplane transformation
                start_3d = workplane._to_3d(primitive.start[0], primitive.start[1])
                end_3d = workplane._to_3d(primitive.end[0], primitive.end[1])

                points = np.array(
                    [
                        [start_3d[0], start_3d[1], start_3d[2]],
                        [end_3d[0], end_3d[1], end_3d[2]],
                    ]
                )
                polyline = pv.Line(points[0], points[1])

                plotter.add_mesh(
                    polyline,
                    color=color,
                    line_width=3,
                    label=label,
                    render_lines_as_tubes=True,
                )

            elif isinstance(primitive, Circle):
                # Sample points around the circle and convert to 3D
                num_points = 100
                theta = np.linspace(0, 2 * np.pi, num_points)
                cx, cy = primitive.center
                r = primitive.radius

                points = []
                for angle in theta:
                    x_2d = cx + r * np.cos(angle)
                    y_2d = cy + r * np.sin(angle)
                    point_3d = workplane._to_3d(x_2d, y_2d)
                    points.append([point_3d[0], point_3d[1], point_3d[2]])

                points_array = np.array(points)
                polyline = pv.Spline(points_array, num_points)

                plotter.add_mesh(
                    polyline,
                    color=color,
                    line_width=3,
                    label=label,
                    render_lines_as_tubes=True,
                )

            elif isinstance(primitive, Arc):
                # Sample points along the arc and convert to 3D
                num_points = 50
                points = []

                for i in range(num_points + 1):
                    t = i / num_points
                    if t < 0.5:
                        t2 = t * 2
                        x_2d = primitive.start[0] * (1 - t2) + primitive.mid[0] * t2
                        y_2d = primitive.start[1] * (1 - t2) + primitive.mid[1] * t2
                    else:
                        t2 = (t - 0.5) * 2
                        x_2d = primitive.mid[0] * (1 - t2) + primitive.end[0] * t2
                        y_2d = primitive.mid[1] * (1 - t2) + primitive.end[1] * t2

                    point_3d = workplane._to_3d(x_2d, y_2d)
                    points.append([point_3d[0], point_3d[1], point_3d[2]])

                points_array = np.array(points)
                polyline = pv.Spline(points_array, num_points)

                plotter.add_mesh(
                    polyline,
                    color=color,
                    line_width=3,
                    label=label,
                    render_lines_as_tubes=True,
                )

    def fea(
        self,
        material: Union["MaterialProperties", str, None] = None,
        loads: Optional[List["Load"]] = None,
        constraints: Optional[List["BoundaryCondition"]] = None,
        mesh_size: float = 2.0,
        element_type: str = "tet4",
    ) -> "FEAResults":
        return self._shapes[0].analyze(
            material=material,
            loads=loads,
            constraints=constraints,
            mesh_size=mesh_size,
            element_type=element_type,
        )

    def to_step(self, file_name: str) -> None:
        """Export all shapes in the app to a single STEP file.

        Args:
            file_name: Path to the output STEP file
        """
        if not self._shapes:
            raise ValueError("No shapes to export")

        # Combine all shapes into one
        combined_shape = self._shapes[0]
        combined_shape.to_step(file_name)