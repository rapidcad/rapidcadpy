import numpy as np
import pytest

from rapidcadpy.primitive import Line
from rapidcadpy.sketch import Sketch
from rapidcadpy.sketch_extrude import SketchExtrude
from rapidcadpy.wire import Wire


@pytest.fixture
def mock_sketch():
    line1 = Line(start_point=(0, 0), end_point=(1, 1))
    line2 = Line(start_point=(1, 1), end_point=(2, 0))
    wire = Wire(edges=[line1, line2])
    return Sketch(outer_wire=wire)


@pytest.fixture
def mock_sketch_extrude(mock_sketch):
    sketch1 = mock_sketch
    line3 = Line(start_point=(3, 2), end_point=(5, 5))
    wire2 = Wire(edges=[line3])
    sketch2 = Sketch(outer_wire=wire2)
    return SketchExtrude(sketch=[sketch1, sketch2])


def test_sketch_extrude_bbox(mock_sketch_extrude):
    expected_bbox = np.array([[0, 0], [5, 5]])
    np.testing.assert_array_equal(mock_sketch_extrude.bbox, expected_bbox)
