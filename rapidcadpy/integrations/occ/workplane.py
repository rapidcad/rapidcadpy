import warnings
from typing import Any, Optional

from OCC.Core.BOPAlgo import BOPAlgo_Tools
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.TopoDS import TopoDS_Compound

from .app import App
from .cad_types import Vector, VectorLike, Vertex
from .integrations.occ.shape import OccShape
from .primitives import Line
from .workplane import Workplane


class OccWorkplane(Workplane):

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
        from .cad_types import Vector

        # Convert to vectors
        origin_vec = Vector(*origin) if not isinstance(origin, Vector) else origin
        normal_vec = Vector(*normal) if not isinstance(normal, Vector) else normal

        # Use default x and y directions for now
        return cls(
            origin=(origin_vec.x, origin_vec.y, origin_vec.z),
            up_dir=(normal_vec.x, normal_vec.y, normal_vec.z),
        )

    def _clear_pending_shapes(self):
        """Clear pending shapes after extrusion to start fresh for next sketch."""
        self._pending_shapes = []
        self._current_position = Vertex(0, 0)
        self._loop_start = None

    def box(
        self, length: float, width: float, height: float, centered: bool = True
    ) -> OccShape:
        """
        Create a 3D box shape.

        Args:
            length: Length of the box (X dimension)
            width: Width of the box (Y dimension)
            height: Height of the box (Z dimension)
            centered: If True (default), box is centered at current position.
                     If False, box extends from current position in positive directions.

        Returns:
            OccShape: The created box shape

        Example:
            # Create a centered box
            box = app.work_plane("XY").box(10, 20, 30)

            # Create a box from origin
            box = app.work_plane("XY").box(10, 20, 30, centered=False)
        """
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        # Determine the starting corner based on centered parameter
        if centered:
            x = self._current_position.x - length / 2
            y = self._current_position.y - width / 2
            z = -height / 2
        else:
            x = self._current_position.x
            y = self._current_position.y
            z = 0

        # Create the box starting point
        corner = gp_Pnt(x, y, z)

        # Create the box
        box_builder = BRepPrimAPI_MakeBox(corner, length, width, height)
        solid = box_builder.Shape()

        # Return as OccShape
        return OccShape(obj=solid, app=self.app)
