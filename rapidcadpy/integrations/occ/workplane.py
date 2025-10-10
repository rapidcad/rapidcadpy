import warnings

from OCP.BOPAlgo import BOPAlgo_Tools
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Pnt, gp_Vec
from OCP.TopoDS import TopoDS_Compound

from rapidcadpy import Vertex, Workplane
from rapidcadpy.integrations.occ.shape import OccShape


class OccWorkplane(Workplane):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def extrude(
        self, distance: float, operation: str = "NewBodyFeatureOperation"
    ) -> "OccShape":
        face = self._make_face()
        extrude_vector = self.z_dir * distance
        extrude_vector = gp_Vec(extrude_vector.x, extrude_vector.y, extrude_vector.z)
        prism_builder: Any = BRepPrimAPI_MakePrism(face, extrude_vector, True)
        return OccShape(obj=prism_builder.Shape())
