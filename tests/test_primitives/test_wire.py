import math

import pytest

from rapidcadpy.cad_types import Vector, Vertex
from rapidcadpy.primitive import Arc, Circle, Line
from rapidcadpy.wire import Wire


# Test: Calculate area of a wire with a line, arc, and circle
def test_cube():
    line1 = Line(
        Vertex(
            0,
            0,
        ),
        Vertex(1, 0),
    )
    line2 = Line(
        Vertex(
            1,
            0,
        ),
        Vertex(1, 1),
    )
    line3 = Line(
        Vertex(
            1,
            1,
        ),
        Vertex(0, 1),
    )
    line4 = Line(
        Vertex(
            0,
            1,
        ),
        Vertex(0, 0),
    )
    cube = Wire([line1, line2, line3, line4])
    assert math.isclose(cube.bounding_box_area(), 1, rel_tol=1e-9)


def test_semi_circle():
    arc = Arc(
        start_point=Vertex(0, 0),
        end_point=Vertex(1, 0),
        center=Vertex(0.5, 0),
        radius=0.5,
        end_angle=math.pi,
        start_angle=0,
    )
    line = Line(Vertex(0, 0), Vertex(1, 0))
    wire = Wire([arc, line])
    assert math.isclose(wire.bounding_box_area(), 0.5, rel_tol=1e-9)


@pytest.mark.xfail(reason="Bounding box of quarter circle arc is not working")
def test_quarter_circle():
    arc = Arc(
        start_point=Vertex(0, 0),
        end_point=Vertex(0.5, 0.5),
        center=Vertex(0.5, 0),
        radius=0.5,
        end_angle=math.pi / 2,
        start_angle=0,
    )
    line1 = Line(Vertex(0, 0), Vertex(0.5, 0))
    line2 = Line(Vertex(0.5, 0), Vertex(0.5, 0.5))
    wire = Wire([arc, line1, line2])
    assert math.isclose(wire.bounding_box_area(), 0.25, rel_tol=1e-9)


def test_pill_shape():
    arc1 = Arc(
        center=Vertex(0, 0.5),
        radius=0.5,
        end_angle=math.pi * 0.5,
        start_angle=0,
        ref_vec=Vector(-1, 0),
    )
    line1 = Line(Vertex(0, 0), Vertex(1, 0))
    arc2 = Arc(
        center=Vertex(1, 0.5),
        radius=0.5,
        end_angle=math.pi * 0.5,
        start_angle=0,
        ref_vec=Vector(0, y=-1),
    )
    line2 = Line(Vertex(1, 1), Vertex(0, 1))

    wire = Wire([line1, arc2, line2, arc1])
    wire.plot()
    assert math.isclose(wire.bounding_box_area(), 2, rel_tol=1e-9)


def test_circle():
    circle = Circle(center=Vertex(0, 0), radius=1)
    wire = Wire([circle])
    assert math.isclose(wire.bounding_box_area(), 4, rel_tol=1e-9)
