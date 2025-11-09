import warnings
from typing import Any, Optional

from OCP.BOPAlgo import BOPAlgo_Tools
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Pnt, gp_Vec
from OCP.TopoDS import TopoDS_Compound

from rapidcadpy.app import App
from rapidcadpy.cad_types import Vector, VectorLike, Vertex
from rapidcadpy.integrations.occ.shape import OccShape
from rapidcadpy.workplane import Workplane


class OccWorkplane(Workplane):
    @classmethod
    def xy_plane(cls, app: Optional[Any] = None) -> "OccWorkplane":
        cls.normal_vector = Vector(0, 0, 1)
        cls.app = app
        return cls(app=app)

    @classmethod
    def xz_plane(cls, app: Optional[Any] = None) -> "OccWorkplane":
        raise NotImplementedError("XZ plane creation not implemented yet.")

    @classmethod
    def yz_plane(cls, app: Optional[Any] = None) -> "OccWorkplane":
        raise NotImplementedError("YZ plane creation not implemented yet.")

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

    def _make_wire(self):
        wire_builder = BRepBuilderAPI_MakeWire()

        from OCP.TopTools import TopTools_ListOfShape

        occ_edges_list = TopTools_ListOfShape()
        for e in self._pending_shapes:
            occ_edges_list.Append(e)
        wire_builder.Add(occ_edges_list)

        wire_builder.Build()

        if not wire_builder.IsDone():
            w = (
                "BRepBuilderAPI_MakeWire::Error(): returns the construction status. BRepBuilderAPI_WireDone if the wire is built, or another value of the BRepBuilderAPI_WireError enumeration indicating why the construction failed = "
                + str(wire_builder.Error())
            )
            warnings.warn(w)

        return wire_builder.Wire()

    def _make_face(self):
        wires = self._make_wire()
        rv = TopoDS_Compound()
        status = BOPAlgo_Tools.WiresToFaces_s(wires, rv)

        if not status:
            raise ValueError("Face construction failed")

        return rv

    def _clear_pending_shapes(self):
        """Clear pending shapes after extrusion to start fresh for next sketch."""
        self._pending_shapes = []
        self._current_position = Vertex(0, 0)
        self._loop_start = None

    def move_to(self, x: float, y: float) -> Workplane:
        self._current_position = Vertex(x, y)
        # Each move_to is treated as the start of a new loop within the sketch
        self._loop_start = Vertex(x, y)
        return self

    def line_to(self, x: float, y: float) -> "OccWorkplane":
        start_point = self._current_position

        start_point_ocp = gp_Pnt(start_point.x, start_point.y, 0)
        end_point_ocp = gp_Pnt(x, y, 0)

        line = BRepBuilderAPI_MakeEdge(start_point_ocp, end_point_ocp).Edge()

        self._pending_shapes.append(line)
        self._current_position = Vertex(x, y)
        return self

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

        # Create rectangle as four lines
        p1 = gp_Pnt(start_x, start_y, 0)
        p2 = gp_Pnt(start_x + width, start_y, 0)
        p3 = gp_Pnt(start_x + width, start_y + height, 0)
        p4 = gp_Pnt(start_x, start_y + height, 0)

        self._pending_shapes.extend(
            [
                BRepBuilderAPI_MakeEdge(p1, p2).Edge(),
                BRepBuilderAPI_MakeEdge(p2, p3).Edge(),
                BRepBuilderAPI_MakeEdge(p3, p4).Edge(),
                BRepBuilderAPI_MakeEdge(p4, p1).Edge(),
            ]
        )

        return self

    def circle(self, radius: float) -> "OccWorkplane":
        """
        Create a circle at the current position.

        Args:
            radius: Radius of the circle

        Returns:
            OccWorkplane: Self for method chaining
        """
        from OCP.gp import gp_Ax2, gp_Circ, gp_Dir

        # Get center point from current position
        center = gp_Pnt(self._current_position.x, self._current_position.y, 0)

        # Create normal direction (z-axis for XY plane)
        normal = gp_Dir(0, 0, 1)

        # Create circle geometry
        circle_gp = gp_Circ(gp_Ax2(center, normal), radius)

        # Create edge from circle
        circle_edge = BRepBuilderAPI_MakeEdge(circle_gp).Edge()

        # Add to pending shapes
        self._pending_shapes.append(circle_edge)

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
        from OCP.GC import GC_MakeArcOfCircle

        # Get the current position as the start point
        start_point = self._current_position

        # Create the three points for the arc
        start_pnt = gp_Pnt(start_point.x, start_point.y, 0)
        mid_pnt = gp_Pnt(p1[0], p1[1], 0)
        end_pnt = gp_Pnt(p2[0], p2[1], 0)

        # Create the arc geometry
        arc_geom = GC_MakeArcOfCircle(start_pnt, mid_pnt, end_pnt).Value()

        # Create the edge from the arc geometry
        arc_edge = BRepBuilderAPI_MakeEdge(arc_geom).Edge()

        # Add to pending shapes
        self._pending_shapes.append(arc_edge)

        # Update current position to the end point
        self._current_position = Vertex(p2[0], p2[1])

        return self

    def extrude(
        self, distance: float, operation: str = "NewBodyFeatureOperation"
    ) -> "OccShape":
        face = self._make_face()
        # Calculate extrude vector
        up_dir_vec = self.normal_vector * distance
        # Convert to gp_Vec using indexing to avoid attribute access issues
        extrude_vector = gp_Vec(
            float(up_dir_vec[0]), float(up_dir_vec[1]), float(up_dir_vec[2])
        )
        prism_builder: Any = BRepPrimAPI_MakePrism(face, extrude_vector, True)

        # Clear pending shapes after extrusion to prepare for next sketch
        self._clear_pending_shapes()

        return OccShape(obj=prism_builder.Shape(), app=self.app)
