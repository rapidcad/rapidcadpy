from typing import Any, Optional


from rapidcadpy.app import App
from rapidcadpy.cad_types import Vector, VectorLike, Vertex
from rapidcadpy.workplane import Workplane
from rapidcadpy.primitives import Line, Circle, Arc


class OccWorkplane(Workplane):
    def __init__(self, app: Optional[Any] = None, *args, **kwargs):
        super().__init__(app=app, *args, **kwargs)
        # Set up coordinate system basis vectors based on normal
        if hasattr(self.__class__, 'normal_vector'):
            self._setup_coordinate_system()
    
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
        if not hasattr(self, '_local_x'):
            self._setup_coordinate_system()
        
        # Transform 2D point to 3D using local coordinate system
        point_3d = self._local_x * x + self._local_y * y
        
        # Apply offset along the normal direction
        offset = getattr(self, '_offset', 0.0)
        offset_3d = self._local_z * offset
        point_3d = point_3d + offset_3d
        
        return (float(point_3d[0]), float(point_3d[1]), float(point_3d[2]))


    @classmethod
    def xy_plane(cls, app: Optional[Any] = None, offset: Optional[float] = None) -> "OccWorkplane":
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
    def xz_plane(cls, app: Optional[Any] = None, offset: Optional[float] = None) -> "OccWorkplane":
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
    def yz_plane(cls, app: Optional[Any] = None, offset: Optional[float] = None) -> "OccWorkplane":
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

    @classmethod
    def create_offset_plane(
        cls, app: App, name: str = "XY", offset: float = 0
    ) -> Workplane:
        raise NotImplementedError("Offset plane creation not implemented yet.")

    @classmethod
    def from_origin_normal(
        cls, origin: VectorLike, normal: VectorLike, app: Optional[Any] = None
    ) -> "OccWorkplane":
        """Create an OccWorkplane from origin and normal vector.

        Args:
            origin: Origin point of the workplane
            normal: Normal vector (up-axis direction)
            app: Optional app instance

        Returns:
            New OccWorkplane with specified origin and normal
        """
        from rapidcadpy.cad_types import Vector

        # Convert to vectors
        origin_vec = Vector(*origin) if not isinstance(origin, Vector) else origin
        normal_vec = Vector(*normal) if not isinstance(normal, Vector) else normal

        # Use default x and y directions for now
        workplane = cls(
            origin=(origin_vec.x, origin_vec.y, origin_vec.z),
            up_dir=(normal_vec.x, normal_vec.y, normal_vec.z),
        )
        if app is not None:
            app.register_workplane(workplane)
        return workplane

    def _clear_pending_shapes(self):
        """Clear pending shapes after extrusion but preserve them in extruded_sketches for visualization."""
        # Preserve the sketch before clearing
        if self._pending_shapes:
            self._extruded_sketches.append(list(self._pending_shapes))
        self._pending_shapes = []
        self._current_position = Vertex(0, 0)
        self._loop_start = None

    def rect(
        self, width: float, height: float, centered: bool = True
    ) -> "OccWorkplane":
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

        self._pending_shapes.extend([
            Line(p1, p2),
            Line(p2, p3),
            Line(p3, p4),
            Line(p4, p1),
        ])

        return self

    def circle(self, radius: float) -> "OccWorkplane":
        """
        Create a circle at the current position.

        Args:
            radius: Radius of the circle

        Returns:
            OccWorkplane: Self for method chaining
        """
        # Store as 2D primitive - conversion to 3D happens in _make_wire
        circle_primitive = Circle(
            center=(self._current_position.x, self._current_position.y),
            radius=radius
        )
        
        self._pending_shapes.append(circle_primitive)
        return self

    def three_point_arc(self, p1: VectorLike, p2: VectorLike) -> "OccWorkplane":
        """
        Create a three-point arc from the current position through p1 to p2.

        Args:
            p1: Middle point of the arc (x, y)
            p2: End point of the arc (x, y)

        Returns:
            OccWorkplane: Self for method chaining
        """
        # Get the current position as the start point
        start_point = self._current_position

        # Store as 2D primitive - conversion to 3D happens in _make_wire
        arc_primitive = Arc(
            start=(start_point.x, start_point.y),
            mid=(p1[0], p1[1]),
            end=(p2[0], p2[1])
        )
        
        self._pending_shapes.append(arc_primitive)

        # Update current position to the end point
        self._current_position = Vertex(p2[0], p2[1])

        return self
