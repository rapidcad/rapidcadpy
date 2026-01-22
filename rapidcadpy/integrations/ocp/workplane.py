from typing import Any, Optional


from rapidcadpy.app import App
from rapidcadpy.cad_types import Vector, VectorLike, Vertex
from rapidcadpy.workplane import Workplane
from rapidcadpy.primitives import Line, Circle, Arc


class OccWorkplane(Workplane):
    def __init__(self, app: Optional[Any] = None, *args, **kwargs):
        super().__init__(app=app, *args, **kwargs)
        # Set up coordinate system basis vectors based on normal
        if hasattr(self.__class__, "normal_vector"):
            self._setup_coordinate_system()

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

        self._pending_shapes.extend(
            [
                Line(p1, p2),
                Line(p2, p3),
                Line(p3, p4),
                Line(p4, p1),
            ]
        )

        return self

    def revolve(
        self,
        angle: float,
        axis: str = "Z",
        operation: str = "NewBodyFeatureOperation",
    ) -> "OccWorkplane":
        """
        Revolve the current sketch around a specified axis.

        Args:
            angle: Angle in degrees to revolve (360 for full revolution)
            axis: Axis to revolve around ('X', 'Y', or 'Z')
            operation: Operation type (default 'NewBodyFeatureOperation')

        Returns:
            OccShape: The revolved 3D solid
        """
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol
        from OCP.gp import gp_Ax1, gp_Pnt, gp_Dir
        from math import radians
        from rapidcadpy.integrations.ocp.sketch2d import OccSketch2D

        # Check if there are shapes to revolve
        if not self._pending_shapes:
            raise ValueError("No shapes to revolve - sketch is empty")

        # Create a Sketch2D from pending shapes to use existing _make_wire and _make_face
        sketch2d = OccSketch2D(
            primitives=self._pending_shapes, workplane=self, app=self.app
        )

        # Build the face to revolve
        face = sketch2d._make_face()

        # Map axis string to direction
        axis_map = {
            "X": gp_Dir(1, 0, 0),
            "Y": gp_Dir(0, 1, 0),
            "Z": gp_Dir(0, 0, 1),
        }

        axis_dir = axis_map.get(axis.upper(), gp_Dir(0, 0, 1))
        axis_origin = gp_Pnt(0, 0, 0)

        # Create the revolve axis
        revolve_axis = gp_Ax1(axis_origin, axis_dir)

        # Perform the revolve operation
        revol_builder = BRepPrimAPI_MakeRevol(face, revolve_axis, radians(angle*360), True)

        # Get the resulting shape
        solid = revol_builder.Shape()

        # Clear pending shapes
        self._clear_pending_shapes()

        # Return OccShape
        from rapidcadpy.integrations.ocp.shape import OccShape

        return OccShape(obj=solid, app=self.app)