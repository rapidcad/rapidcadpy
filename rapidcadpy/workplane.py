"""
Workplane module - Provides a CadQuery-like fluent API for CAD operations
and coordinate system representation (unified Plane functionality).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from rapidcadpy.primitives import Arc, Circle, Line

from .cad_types import Vector, VectorLike, Vertex

if TYPE_CHECKING:
    from rapidcadpy.app import App
    from rapidcadpy.sketch2d import Sketch2D


class Workplane(ABC):
    """
    A unified class that provides both coordinate system representation (plane functionality)
    and a fluent API for CAD operations.

    This class maintains a working coordinate system and allows chaining operations to build
    complex 3D models using a simple, intuitive syntax.
    """

    def __init__(
        self,
        app: Optional["App"] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize a workplane with coordinate system and working state.

        Args:
            origin: Origin point of the coordinate system
            x_dir: X-axis direction vector (ignored if theta/phi/gamma provided)
            y_dir: Y-axis direction vector (ignored if theta/phi/gamma provided)
            up_dir: Up-axis direction vector (ignored if theta/phi/gamma provided)
            theta: Legacy PlaneOld API - rotation angle around z-axis
            phi: Legacy PlaneOld API - rotation angle around y-axis
            gamma: Legacy PlaneOld API - rotation angle around x-axis
        """

        # Working state (fluent API functionality)
        from .cad_types import Vertex

        self._current_position = Vertex(0, 0)
        self._pending_shapes = []
        self._extruded_sketches = []  # Preserve sketches even after extrusion
        self._current_sketch = None
        self._loop_start: Vertex = Vertex(
            0, 0
        )  # Track the start of the current sketch loop
        self._offset = 0.0  # Default offset
        self.app = app

        if hasattr(self.__class__, "normal_vector"):
            self._setup_coordinate_system()

    # ========== Factory methods for standard planes ==========

    @classmethod
    def xy_plane(
        cls, app: Optional[Any] = None, offset: Optional[float] = None
    ) -> "Workplane":
        cls.normal_vector = Vector(0, 0, 1)
        cls.app = app
        workplane = cls(app=app)

        # Apply offset if provided (offset is along the normal direction)
        if offset is not None:
            workplane._offset = offset
        else:
            workplane._offset = 0.0

        if app is not None:
            app.register_workplane(workplane)
        return workplane

    @classmethod
    def xz_plane(
        cls, app: Optional[Any] = None, offset: Optional[float] = None
    ) -> "Workplane":
        cls.normal_vector = Vector(0, 1, 0)
        cls.app = app
        workplane = cls(app=app)

        # Apply offset if provided (offset is along the normal direction)
        if offset is not None:
            workplane._offset = offset
        else:
            workplane._offset = 0.0

        if app is not None:
            app.register_workplane(workplane)
        return workplane

    @classmethod
    def yz_plane(
        cls, app: Optional[Any] = None, offset: Optional[float] = None
    ) -> "Workplane":
        cls.normal_vector = Vector(1, 0, 0)
        cls.app = app
        workplane = cls(app=app)

        # Apply offset if provided (offset is along the normal direction)
        if offset is not None:
            workplane._offset = offset
        else:
            workplane._offset = 0.0

        if app is not None:
            app.register_workplane(workplane)
        return workplane

    @abstractmethod
    def create_offset_plane(
        cls, app: "App", name: str = "XY", offset: float = 0.0
    ) -> "Workplane":
        """Create a standard named workplane with an offset.

        Args:
            name: Standard plane name ("XY", "XZ", "YZ")
            offset: Offset distance from the standard plane
            app: Optional app instance
        Returns:
            Workplane instance at the specified offset
        """
        ...

    @classmethod
    def from_origin_normal(
        cls,
        app: "App",
        origin: VectorLike,
        normal: VectorLike,
    ) -> "Workplane":
        """Create a workplane from origin and normal vector.

        Args:
            origin: Origin point of the workplane
            normal: Normal vector (z-axis direction)
            app: Optional app instance

        Returns:
            New workplane with specified origin and normal
        """
        from rapidcadpy.cad_types import Vector

        # Convert to vectors
        origin_vec = Vector(*origin) if not isinstance(origin, Vector) else origin
        normal_vec = Vector(*normal) if not isinstance(normal, Vector) else normal

        # Use default x and y directions for now
        return cls(
            origin=origin_vec,
            up_dir=normal_vec,
            app=app,
        )

    @classmethod
    def from_n_x_axis(
        cls, origin: VectorLike, normal: VectorLike, x_axis: VectorLike
    ) -> "Workplane":
        """Create a workplane from origin, normal vector, and x-axis vector.

        This method provides compatibility with the old PlaneOld.from_n_x_axis method.
        """
        import numpy as np

        from rapidcadpy.cad_types import Vector

        # Convert to vectors
        origin_vec = Vector(*origin) if not isinstance(origin, Vector) else origin
        normal_vec = Vector(*normal) if not isinstance(normal, Vector) else normal
        x_axis_vec = Vector(*x_axis) if not isinstance(x_axis, Vector) else x_axis

        # Normalize vectors
        normal_normalized = normal_vec.normalize()
        x_axis_normalized = x_axis_vec.normalize()

        # Calculate y-axis as cross product of normal and x-axis using numpy
        y_cross = np.cross(normal_normalized, x_axis_normalized)
        y_axis_normalized = Vector(y_cross[0], y_cross[1], y_cross[2]).normalize()

        return cls(
            origin=origin_vec,
            x_dir=Vector(
                x_axis_normalized[0], x_axis_normalized[1], x_axis_normalized[2]
            ),
            y_dir=Vector(
                y_axis_normalized[0], y_axis_normalized[1], y_axis_normalized[2]
            ),
            up_dir=Vector(
                normal_normalized[0], normal_normalized[1], normal_normalized[2]
            ),
        )

    @classmethod
    def from_n_x_y_axes(
        cls,
        origin: VectorLike,
        normal: VectorLike,
        x_axis: VectorLike,
        y_axis: VectorLike,
    ) -> "Workplane":
        """Create a workplane from origin, normal vector, x-axis vector, and y-axis vector.

        This method provides compatibility with the old PlaneOld.from_n_x_y_axes method.
        """
        from rapidcadpy.cad_types import Vector

        # Convert to vectors and normalize
        origin_vec = Vector(*origin) if not isinstance(origin, Vector) else origin
        normal_vec = Vector(*normal) if not isinstance(normal, Vector) else normal
        x_axis_vec = Vector(*x_axis) if not isinstance(x_axis, Vector) else x_axis
        y_axis_vec = Vector(*y_axis) if not isinstance(y_axis, Vector) else y_axis

        # Normalize and convert back to Vector
        x_norm = x_axis_vec.normalize()
        y_norm = y_axis_vec.normalize()
        z_norm = normal_vec.normalize()

        return cls(
            origin=origin_vec,
            x_dir=Vector(x_norm[0], x_norm[1], x_norm[2]),
            y_dir=Vector(y_norm[0], y_norm[1], y_norm[2]),
            up_dir=Vector(z_norm[0], z_norm[1], z_norm[2]),
        )

    @property
    def normal(self) -> Vector:
        """Get the normal vector of the plane (up_dir)."""
        return self.up_dir

    def translate_plane(self, offset: VectorLike) -> "Workplane":
        """
        Create a new workplane with coordinate system translated by the given offset.

        Args:
            offset: Translation vector

        Returns:
            New workplane with translated coordinate system
        """
        if isinstance(offset, Vector):
            new_origin = Vector(
                self.origin.x + offset.x,
                self.origin.y + offset.y,
                self.origin.z + offset.z,
            )
        else:
            offset_vec = Vector(*offset)
            new_origin = Vector(
                self.origin.x + offset_vec.x,
                self.origin.y + offset_vec.y,
                self.origin.z + offset_vec.z,
            )

        new_workplane = Workplane(new_origin, self.x_dir, self.y_dir, self.up_dir)
        new_workplane._pending_shapes = self._pending_shapes.copy()
        new_workplane._current_position = self._current_position
        return new_workplane

    def round(self, decimals=6):
        """Round coordinate system values to specified decimal places."""
        self.origin = Vector(
            round(self.origin.x, decimals),
            round(self.origin.y, decimals),
            round(self.origin.z, decimals),
        )
        self.x_dir = Vector(
            round(self.x_dir.x, decimals),
            round(self.x_dir.y, decimals),
            round(self.x_dir.z, decimals),
        )
        self.y_dir = Vector(
            round(self.y_dir.x, decimals),
            round(self.y_dir.y, decimals),
            round(self.y_dir.z, decimals),
        )
        self.z_dir = Vector(
            round(self.z_dir.x, decimals),
            round(self.z_dir.y, decimals),
            round(self.z_dir.z, decimals),
        )

    def _setup_coordinate_system(self):
        """Set up the local coordinate system based on the workplane normal."""
        normal = self.normal_vector

        # Define local X and Y axes based on the normal
        if abs(normal.z) > 0.9:  # XY plane (normal along Z)
            self._local_x = Vector(1, 0, 0)
            self._local_y = Vector(0, 1, 0)
        elif abs(normal.y) > 0.9:  # XZ plane (normal along Y)
            self._local_x = Vector(1, 0, 0)
            self._local_y = Vector(0, 0, 1)
        else:  # YZ plane (normal along X)
            self._local_x = Vector(0, 1, 0)
            self._local_y = Vector(0, 0, 1)

        self._local_z = normal

    def _to_3d(self, x: float, y: float) -> tuple[float, float, float]:
        """Convert 2D workplane coordinates to 3D world coordinates.

        Args:
            x: X coordinate in the workplane
            y: Y coordinate in the workplane

        Returns:
            Tuple of (x, y, z) in 3D world coordinates
        """
        if not hasattr(self, "_local_x"):
            self._setup_coordinate_system()

        # Transform 2D point to 3D using local coordinate system
        point_3d = self._local_x * x + self._local_y * y

        # Apply offset along the normal direction
        offset = getattr(self, "_offset", 0.0)
        offset_3d = self._local_z * offset
        point_3d = point_3d + offset_3d

        return (float(point_3d[0]), float(point_3d[1]), float(point_3d[2]))

    def line_to(self, x: float, y: float) -> "Workplane":
        start_point = self._current_position

        # Store as 2D primitive - conversion to 3D happens in _make_wire
        line_primitive = Line(start=(start_point.x, start_point.y), end=(x, y))

        self._pending_shapes.append(line_primitive)
        self._current_position = Vertex(x, y)
        return self

    def move_to(self, x: float, y: float) -> "Workplane":
        """
        Move the current position to the specified coordinates.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Self for method chaining
        """
        self._current_position = Vertex(x, y)
        # Each move_to is treated as the start of a new loop within the sketch
        self._loop_start = Vertex(x, y)
        return self

    def circle(self, radius: float) -> "Workplane":
        """
        Create a circle at the current position.

        Args:
            radius: Radius of the circle

        Returns:
            Workplane: Self for method chaining
        """
        # Store as 2D primitive - conversion to 3D happens in _make_wire
        circle_primitive = Circle(
            center=(self._current_position.x, self._current_position.y), radius=radius
        )

        self._pending_shapes.append(circle_primitive)
        return self

    def three_point_arc(self, p1: VectorLike, p2: VectorLike) -> "Workplane":
        """
        Create a three-point arc from the current position through p1 to p2.

        Args:
            p1: Middle point of the arc (x, y)
            p2: End point of the arc (x, y)

        Returns:
            Workplane: Self for method chaining
        """
        # Get the current position as the start point
        start_point = self._current_position

        # Store as 2D primitive - conversion to 3D happens in _make_wire
        arc_primitive = Arc(
            start=(start_point.x, start_point.y), mid=(p1[0], p1[1]), end=(p2[0], p2[1])
        )

        self._pending_shapes.append(arc_primitive)

        # Update current position to the end point
        self._current_position = Vertex(p2[0], p2[1])

        return self

    def close(self) -> "Sketch2D":
        """
        Close the current sketch and construct the face.

        This finalizes the sketch by creating a face from the pending edges
        and returns a Sketch2D object that can be extruded.

        Returns:
            Sketch2D: A sketch object containing the constructed face
        """

        # add a line from current position to loop start if they are not the same
        if len(self._pending_shapes) == 0:
            raise ValueError("No Primitives to Create a Sketch")
        if (
            abs(self._current_position.x - self._loop_start.x) > 1e-9
            or abs(self._current_position.y - self._loop_start.y) > 1e-9
        ):
            self.line_to(self._loop_start.x, self._loop_start.y)

        # Create the Sketch2D object
        sketch = self.app.sketch_class(
            primitives=self._pending_shapes, workplane=self, app=self.app
        )
        # Preserve the sketch edges for visualization
        self._clear_pending_shapes()

        # Reset current position
        self._current_position = Vertex(0, 0)

        return sketch

    def finish(self) -> "Sketch2D":
        """
        Finalize the current sketch as an open wire/path without closing it.

        This is useful for creating pipes along open paths where you don't want
        the path to automatically close back to the starting point.

        Returns:
            Sketch2D: A sketch object containing the open wire

        Example:
            # Create a pipe along a diagonal line
            wp = app.work_plane("XY")
            diagonal = wp.move_to(0, 0).line_to(100, 50).as_wire()
            pipe = diagonal.pipe(diameter=10)
        """
        if len(self._pending_shapes) == 0:
            raise ValueError("No Primitives to Create a Wire")

        # Create the Sketch2D object WITHOUT closing the loop
        sketch = self.app.sketch_class(
            primitives=self._pending_shapes, workplane=self, app=self.app
        )
        # Preserve the sketch edges for visualization
        self._clear_pending_shapes()

        # Reset current position
        self._current_position = Vertex(0, 0)

        return sketch

    def rect(self, width: float, height: float, centered: bool = True) -> "Workplane":
        """
        Create a rectangle at the current position.

        Args:
            width: Rectangle width
            height: Rectangle height
            centered: Whether to center the rectangle around the current position

        Returns:
            Self for method chaining
        """
        if centered:
            x_offset = width / 2
            y_offset = height / 2
            start_x = self._current_position.x - x_offset
            start_y = self._current_position.y - y_offset
        else:
            start_x = self._current_position.x
            start_y = self._current_position.y

        # Create rectangle as four 2D line primitives
        p1 = (start_x, start_y)
        p2 = (start_x + width, start_y)
        p3 = (start_x + width, start_y + height)
        p4 = (start_x, start_y + height)

        self._pending_shapes.extend(
            [
                Line(p1, p2),
                Line(p2, p3),
                Line(p3, p4),
                Line(p4, p1),
            ]
        )

        return self

    def rotate(self, axis: VectorLike, angle: float) -> "Workplane":
        """
        Rotate the workplane's coordinate system around a given axis by a specified angle (in radians).

        Args:
            axis: Axis to rotate around (as a 3D vector)
            angle: Rotation angle in radians

        Returns:
            New Workplane with rotated coordinate system
        """
        import numpy as np

        from rapidcadpy.cad_types import Vector

        # Normalize axis
        axis_vec = Vector(*axis) if not isinstance(axis, Vector) else axis
        axis_np = np.array([axis_vec.x, axis_vec.y, axis_vec.z])
        axis_np = axis_np / np.linalg.norm(axis_np)

        # Rotation matrix (Rodrigues' formula)
        K = np.array(
            [
                [0, -axis_np[2], axis_np[1]],
                [axis_np[2], 0, -axis_np[0]],
                [-axis_np[1], axis_np[0], 0],
            ]
        )
        I = np.eye(3)
        R = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        def rotate_vec(vec):
            v_np = np.array([vec.x, vec.y, vec.z])
            v_rot = R @ v_np
            return Vector(*v_rot)

        new_x = rotate_vec(self.x_dir)
        new_y = rotate_vec(self.y_dir)
        new_z = rotate_vec(self.z_dir)

        # Return new Workplane with rotated axes
        wp = Workplane(self.origin, new_x, new_y, new_z)
        wp._pending_shapes = self._pending_shapes.copy()
        wp._current_position = self._current_position
        return wp

    def extrude(self, distance: float):
        """
        Alias function for sketch = wp.close(); sketch.extrude()
        """
        if len(self._pending_shapes) == 0:
            raise ValueError("No Pending Primitves to Extrude")
        sketch = self.close()
        extruded_shape = sketch.extrude(distance)
        return extruded_shape

    def to_png(
        self,
        file_name: Optional[str] = None,
        width: int = 800,
        height: int = 600,
        margin: float = 100,
    ) -> None:
        """
        Render the current 2D sketch to a PNG image.

        Args:
            file_name: Path to save the PNG file. If None, displays in a UI window instead.
            width: Image width in pixels (default: 800)
            height: Image height in pixels (default: 600)
            margin: Margin around the sketch in pixels (default: 100)

        Raises:
            ValueError: If no shapes are in the sketch
            ImportError: If matplotlib is not installed
        """
        if not self._pending_shapes and not self._extruded_sketches:
            raise ValueError("No shapes to render in sketch")

        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle as MplCircle
            import numpy as np
        except ImportError:
            raise ImportError(
                "matplotlib is required for sketch rendering. Install with: pip install matplotlib"
            )

        # Create figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.set_aspect("equal")

        # Collect all points to calculate bounds
        all_points = []

        # Render current pending primitives
        for primitive in self._pending_shapes:
            if isinstance(primitive, Line):
                xs = [primitive.start[0], primitive.end[0]]
                ys = [primitive.start[1], primitive.end[1]]
                ax.plot(xs, ys, "k-", linewidth=2)
                all_points.extend([primitive.start, primitive.end])

            elif isinstance(primitive, Circle):
                circ = MplCircle(
                    primitive.center,
                    primitive.radius,
                    fill=False,
                    edgecolor="black",
                    linewidth=2,
                )
                ax.add_patch(circ)
                # Add bounding points
                cx, cy = primitive.center
                r = primitive.radius
                all_points.extend(
                    [
                        (cx - r, cy),
                        (cx + r, cy),
                        (cx, cy - r),
                        (cx, cy + r),
                    ]
                )

            elif isinstance(primitive, Arc):
                # For arcs, sample points - use simple linear interpolation for visualization
                num_points = 50
                arc_x = []
                arc_y = []

                for i in range(num_points + 1):
                    t = i / num_points
                    if t < 0.5:
                        # First half: start to mid
                        t2 = t * 2
                        x = primitive.start[0] * (1 - t2) + primitive.mid[0] * t2
                        y = primitive.start[1] * (1 - t2) + primitive.mid[1] * t2
                    else:
                        # Second half: mid to end
                        t2 = (t - 0.5) * 2
                        x = primitive.mid[0] * (1 - t2) + primitive.end[0] * t2
                        y = primitive.mid[1] * (1 - t2) + primitive.end[1] * t2
                    arc_x.append(x)
                    arc_y.append(y)
                    all_points.append((x, y))

                ax.plot(arc_x, arc_y, "k-", linewidth=2)

        # Render extruded sketches (these now contain primitives too)
        for primitive_list in self._extruded_sketches:
            for primitive in primitive_list:
                if isinstance(primitive, Line):
                    xs = [primitive.start[0], primitive.end[0]]
                    ys = [primitive.start[1], primitive.end[1]]
                    ax.plot(xs, ys, "k-", linewidth=2)
                    all_points.extend([primitive.start, primitive.end])

                elif isinstance(primitive, Circle):
                    circ = MplCircle(
                        primitive.center,
                        primitive.radius,
                        fill=False,
                        edgecolor="black",
                        linewidth=2,
                    )
                    ax.add_patch(circ)
                    cx, cy = primitive.center
                    r = primitive.radius
                    all_points.extend(
                        [
                            (cx - r, cy),
                            (cx + r, cy),
                            (cx, cy - r),
                            (cx, cy + r),
                        ]
                    )

                elif isinstance(primitive, Arc):
                    num_points = 50
                    arc_x = []
                    arc_y = []

                    for i in range(num_points + 1):
                        t = i / num_points
                        if t < 0.5:
                            t2 = t * 2
                            x = primitive.start[0] * (1 - t2) + primitive.mid[0] * t2
                            y = primitive.start[1] * (1 - t2) + primitive.mid[1] * t2
                        else:
                            t2 = (t - 0.5) * 2
                            x = primitive.mid[0] * (1 - t2) + primitive.end[0] * t2
                            y = primitive.mid[1] * (1 - t2) + primitive.end[1] * t2
                        arc_x.append(x)
                        arc_y.append(y)
                        all_points.append((x, y))

                    ax.plot(arc_x, arc_y, "k-", linewidth=2)

        # Calculate bounds with margin
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Add margin
            x_range = max(max_x - min_x, 1)
            y_range = max(max_y - min_y, 1)
            margin_x = x_range + margin
            margin_y = y_range + margin

            ax.set_xlim(min_x - margin_x, max_x + margin_x)
            ax.set_ylim(min_y - margin_y, max_y + margin_y)

        # Determine axis labels based on workplane normal
        # The normal vector determines which 2D plane we're working in
        if hasattr(self, "normal_vector"):
            normal = self.normal_vector
            # XY plane: normal is (0, 0, 1) - axes are X and Y
            if abs(normal.z) > 0.9:
                x_label = "X"
                y_label = "Y"
            # XZ plane: normal is (0, 1, 0) - axes are X and Z
            elif abs(normal.y) > 0.9:
                x_label = "X"
                y_label = "Z"
            # YZ plane: normal is (1, 0, 0) - axes are Y and Z
            elif abs(normal.x) > 0.9:
                x_label = "Y"
                y_label = "Z"
            else:
                # Generic case
                x_label = "U"
                y_label = "V"
        else:
            # Default labels if no normal vector is set
            x_label = "X"
            y_label = "Y"

        # Style
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("2D Sketch")

        # Save or show
        plt.tight_layout()
        if file_name:
            plt.savefig(file_name, dpi=100, bbox_inches="tight", facecolor="white")
            plt.close()
        else:
            # Display in interactive window
            plt.show()
