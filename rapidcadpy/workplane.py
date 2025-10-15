"""
Workplane module - Provides a CadQuery-like fluent API for CAD operations
and coordinate system representation (unified Plane functionality).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from rapidcadpy.shape import Shape

from .cad_types import Vector, VectorLike

if TYPE_CHECKING:
    from rapidcadpy.app import App


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
        self._current_sketch = None
        self.app = app

    # ========== Factory methods for standard planes ==========

    @abstractmethod
    def xy_plane(
        cls, app: "App", *args, **kwargs
    ) -> "Workplane":
        """Create a workplane in the XY orientation at the given origin."""
        ...

    @abstractmethod
    def xz_plane(
        cls, app:"App", *args, **kwargs
    ) -> "Workplane":
        """Create a workplane in the XZ orientation at the given origin."""
        ...

    @abstractmethod
    def yz_plane(
        cls, app: "App", *args, **kwargs
    ) -> "Workplane":
        """Create a workplane in the YZ orientation at the given origin."""
        ...
    
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
        cls, app: "App", origin: VectorLike, normal: VectorLike,
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

    # ========== Coordinate system properties and methods ==========

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

    @staticmethod
    def from_dict(transform_dict: Dict[str, Any]) -> "Workplane":
        """Create a workplane from a dictionary representation."""
        origin = Vector(
            transform_dict["origin"]["x"],
            transform_dict["origin"]["y"],
            transform_dict["origin"]["z"],
        )

        x_axis = Vector(
            transform_dict["x_axis"]["x"],
            transform_dict["x_axis"]["y"],
            transform_dict["x_axis"]["z"],
        )

        y_axis = Vector(
            transform_dict["y_axis"]["x"],
            transform_dict["y_axis"]["y"],
            transform_dict["y_axis"]["z"],
        )

        z_axis = Vector(
            transform_dict["z_axis"]["x"],
            transform_dict["z_axis"]["y"],
            transform_dict["z_axis"]["z"],
        )

        return Workplane(origin, x_axis, y_axis, z_axis)

    def to_json(self) -> Dict[str, Any]:
        """Convert coordinate system to JSON representation."""
        return {
            "origin": self.origin.to_json(),
            "x_dir": {
                "x": float(self.x_dir.x),
                "y": float(self.x_dir.y),
                "z": float(self.x_dir.z),
            },
            "y_dir": {
                "x": float(self.y_dir.x),
                "y": float(self.y_dir.y),
                "z": float(self.y_dir.z),
            },
            "up_dir": {
                "x": float(self.up_dir.x),
                "y": float(self.up_dir.y),
                "z": float(self.up_dir.z),
            },
        }

    @staticmethod
    def from_json(json_data: Dict[str, Any]) -> "Workplane":
        """Create a workplane from JSON representation."""
        origin = Vector.from_json(json_data["origin"])
        x_dir = Vector.from_json(json_data["x_dir"])
        y_dir = Vector.from_json(json_data["y_dir"])
        up_dir = Vector.from_json(json_data["up_dir"])
        return Workplane(origin, x_dir, y_dir, up_dir)

    def to_python(self):
        """Generate Python code representation."""
        return f"Workplane(origin={self.origin.to_python()}, x_dir={self.x_dir.to_python()}, y_dir={self.y_dir.to_python()}, up_dir={self.up_dir.to_python()})"

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

    # ========== Fluent API methods ==========

    @abstractmethod
    def circle(self, radius: float) -> "Workplane":
        """
        Create a circle at the current position.

        Args:
            radius: Circle radius

        Returns:
            Self for method chaining
        """
        ...

    @abstractmethod
    def line_to(self, x: float, y: float) -> "Workplane": ...

    def move_to(self, x: float, y: float) -> "Workplane":
        """
        Move the current position to the specified coordinates.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Self for method chaining
        """
        from .cad_types import Vertex

        self._current_position = Vertex(x, y)
        return self

    @abstractmethod
    def three_point_arc(self, p1: VectorLike, p2: VectorLike) -> "Workplane":
        """
        Create a three-point arc from the current position through p1 to p2.

        Args:
            p1: First point on the arc
            p2: Second point on the arc
        """
        ...

    @abstractmethod
    def rect(
        self, width: float, height: float, centered: bool = True
    ) -> "Workplane": ...

    @abstractmethod
    def extrude(
        self, distance: float, operation: str = "NewBodyFeatureOperation"
    ) -> Shape: ...

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
