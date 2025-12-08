import warnings
from typing import Any, Optional

from OCC.Core.BOPAlgo import BOPAlgo_Tools
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.TopoDS import TopoDS_Compound

from rapidcadpy.app import App
from rapidcadpy.cad_types import Vector, VectorLike, Vertex
from rapidcadpy.integrations.occ.shape import OccShape
from rapidcadpy.primitives import Line
from rapidcadpy.workplane import Workplane


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
        from rapidcadpy.cad_types import Vector

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
