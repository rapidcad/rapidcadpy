from rapidcadpy.cad_types import Vector, Vertex
from rapidcadpy.cadseq import Cad, SketchExtrude, Wire
from rapidcadpy.extrude import Extrude
from rapidcadpy.primitive import Circle, Line
from rapidcadpy.sketch import Sketch
from rapidcadpy.workplane import Workplane as PlaneOld

vertex_0 = Vertex(x=2.0, y=-2.0)
vertex_1 = Vertex(x=0.0, y=-2.0)
vertex_2 = Vertex(x=2.0, y=0.0)
vertex_3 = Vertex(x=0.0, y=0.0)
vertex_4 = Vertex(x=1.5, y=-1.5)
vertex_5 = Vertex(x=1.0, y=-0.8)
edge_0 = Line(start_point=vertex_0, end_point=vertex_1)
edge_1 = Line(start_point=vertex_2, end_point=vertex_0)
edge_2 = Line(start_point=vertex_3, end_point=vertex_2)
edge_3 = Line(start_point=vertex_1, end_point=vertex_3)
circle_1 = Circle(center=vertex_4, radius=0.125)
circle_0 = Circle(center=vertex_5, radius=0.25)
outer_wire_0 = Wire([edge_0, edge_1, edge_2, edge_3])
inner_wire_0 = Wire([circle_1])
sketch_0 = Sketch(
    outer_wire=outer_wire_0,
    inner_wires=[inner_wire_0],
    sketch_plane=PlaneOld(
        origin=Vector(0.0, 0.0, 0.0), theta=1.570796, phi=1.570796, gamma=-3.141593
    ),
)
extrude_0 = Extrude(
    extent_one=1.2,
    extent_two=0.0,
    symmetric=False,
    taper_angle_one=0.0,
    taper_angle_two=0.0,
)
sketch_extrude_0 = SketchExtrude(sketch_0, extrude_0)

vertex_0 = Vertex(x=1.5, y=-1.5)
circle = Circle(center=vertex_0, radius=0.125)
outer_wire_1 = Wire([circle])

sketch_1 = Sketch(
    outer_wire=outer_wire_1,
    sketch_plane=PlaneOld(
        origin=Vector(0.0, 0.0, 0.0), theta=1.570796, phi=1.570796, gamma=-3.141593
    ),
)
extrude_1 = Extrude(
    extent_one=1.2,
    extent_two=0.0,
    symmetric=False,
    taper_angle_one=0.0,
    taper_angle_two=0.0,
)
sketch_extrude_1 = SketchExtrude(sketch_1, extrude_1)

vertex_0 = Vertex(x=1.0, y=-0.8)
circle = Circle(center=vertex_0, radius=0.25)
outer_wire_2 = Wire([circle])

sketch_2 = Sketch(
    outer_wire=outer_wire_2,
    sketch_plane=PlaneOld(
        origin=Vector(0.0, 0.0, 0.0), theta=1.570796, phi=1.570796, gamma=-3.141593
    ),
)
extrude_2 = Extrude(
    extent_one=1.2,
    extent_two=0.0,
    symmetric=False,
    taper_angle_one=0.0,
    taper_angle_two=0.0,
)
sketch_extrude_2 = SketchExtrude(sketch_2, extrude_2)

vertex_0 = Vertex(x=0.697262, y=0.6)
circle = Circle(center=vertex_0, radius=0.100037)
outer_wire_3 = Wire([circle])

sketch_3 = Sketch(
    outer_wire=outer_wire_3,
    sketch_plane=PlaneOld(
        origin=Vector(0.0, 0.0, 0.0), theta=1.570796, phi=-0.0, gamma=1.570796
    ),
)
extrude_3 = Extrude(
    extent_one=-0.5,
    extent_two=0.0,
    symmetric=False,
    taper_angle_one=0.0,
    taper_angle_two=0.0,
)
sketch_extrude_3 = SketchExtrude(sketch_3, extrude_3)

vertex_0 = Vertex(x=1.5, y=0.6)
circle = Circle(center=vertex_0, radius=0.3)
outer_wire_4 = Wire([circle])

sketch_4 = Sketch(
    outer_wire=outer_wire_4,
    sketch_plane=PlaneOld(
        origin=Vector(0, 0, 0), theta=1.570796, phi=-0.0, gamma=1.570796
    ),
)
extrude_4 = Extrude(
    extent_one=-2.2,
    extent_two=0.0,
    symmetric=False,
    taper_angle_one=0.0,
    taper_angle_two=0.0,
)
sketch_extrude_4 = SketchExtrude(sketch_4, extrude_4)

vertex_0 = Vertex(x=1.0, y=-0.8)
circle = Circle(center=vertex_0, radius=0.25)
outer_wire_5 = Wire([circle])

sketch_5 = Sketch(
    outer_wire=outer_wire_5,
    sketch_plane=PlaneOld(
        origin=Vector(0.0, -0.6, 1.0), theta=1.570796, phi=-0.0, gamma=1.570796
    ),
)
extrude_5 = Extrude(
    extent_one=1.2,
    extent_two=0.0,
    symmetric=False,
    taper_angle_one=0.0,
    taper_angle_two=0.0,
)
sketch_extrude_5 = SketchExtrude(sketch_5, extrude_5)

vertex_0 = Vertex(x=1.5, y=-1.5)
circle = Circle(center=vertex_0, radius=0.125)
outer_wire_6 = Wire([circle])

sketch_6 = Sketch(
    outer_wire=outer_wire_6,
    sketch_plane=PlaneOld(
        origin=Vector(0.0, 0.0, 0.0), theta=1.570796, phi=-0.0, gamma=1.570796
    ),
)
extrude_6 = Extrude(
    extent_one=0.5,
    extent_two=0.0,
    symmetric=False,
    taper_angle_one=0.0,
    taper_angle_two=0.0,
)
sketch_extrude_6 = SketchExtrude(sketch_6, extrude_6)

cad_seq = Cad(
    [
        sketch_extrude_0,
        # sketch_extrude_1,
        # sketch_extrude_2,
        # sketch_extrude_3,
        sketch_extrude_4,
        # sketch_extrude_5,
        # sketch_extrude_6,
    ]
)

cad_seq.render_3d()
