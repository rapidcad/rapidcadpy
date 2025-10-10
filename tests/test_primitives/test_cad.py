import json
import math
import os
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from rapidcadpy.cad import Cad
from rapidcadpy.cad_types import Vertex
from rapidcadpy.json_importer.process_deepcad import DeepCadJsonParser
from rapidcadpy.json_importer.process_f360 import Fusion360GalleryParser
from rapidcadpy.functions import visualize_lines
from rapidcadpy.primitive import Arc, Circle, Line

DATA_ROOT = "/Users/elias.berger/cadgpt/data"

DATA_FOLDER = pathlib.Path(DATA_ROOT).joinpath("deepcad_json")


@pytest.fixture
def cad_sequence_1():
    json_file_path = os.path.join(DATA_FOLDER, "0000/00007848.json")
    return DeepCadJsonParser.process(json_file_path)


@pytest.fixture
def random_cad_sequence():
    random_file = np.random.choice(list(DATA_FOLDER.glob("*.json")))
    print(f"Random file: {random_file}")
    return DeepCadJsonParser.process(random_file)


@pytest.fixture
def random_py_file():
    random_file = np.random.choice(list(PY_FOLDER.glob("*.py")))
    with open(random_file, "r") as file:
        code = file.read()
    return code


def cad_from_id(id: str) -> Any:
    json_file_path = os.path.join(DATA_FOLDER, f"{id}.json")
    return DeepCadJsonParser.process(json_file_path)


@pytest.fixture
def internal_circle_cad():
    json_file_path = os.path.join(DATA_FOLDER, "0000/00007848.json")
    return DeepCadJsonParser.process(json_file_path)


@pytest.fixture
def flash_cad() -> Cad:
    json_file_path = os.path.join(DATA_FOLDER, "0056/00561068.json")
    return DeepCadJsonParser.process(json_file_path)


@pytest.fixture
def two_dimensions_cad_sequence():
    json_file_path = os.path.join(DATA_FOLDER, "0066/00665265.json")
    return DeepCadJsonParser.process(json_file_path)


@pytest.fixture
def long_sequence():
    json_file_path = os.path.join(DATA_FOLDER, "0095/00956264.json")
    return DeepCadJsonParser.process(json_file_path)


@pytest.fixture
def long_sequence_omni():
    OMNI_CAD_DATA_FOLDER = pathlib.Path(__file__).parent.parent.parent.joinpath(
        "data", "omni_cad", "Omni-CAD", "json"
    )
    json_file_path = os.path.join(OMNI_CAD_DATA_FOLDER, "0095/00953100_00001.json")
    return DeepCadJsonParser.process(json_file_path)


@pytest.fixture
def shield_cad():
    json_file_path = os.path.join(DATA_FOLDER, "0095/00950156.json")
    return DeepCadJsonParser.process(json_file_path)


def test_rapidcadpy_import(flash_cad):
    # two_dimensions_cad_sequence.show_3d()
    flash_cad.normalize()
    flash_cad.to_stl("flash_cad.stl")
    # two_dimensions_cad_sequence.show_3d()
    flash_cad.round(decimals=2)
    print(flash_cad.to_python())


class TestParseEdge:
    def test_parse_edge_line(self, cad_sequence):
        start_point = Vertex(x=0, y=0)
        end_point = Vertex(x=1, y=1)
        line = Line(start_point=start_point, end_point=end_point)

        # Parse the edge and verify geometry
        result_edge = cad_sequence._parse_edge(line)
        assert isinstance(result_edge, Build123dEdge)
        assert np.allclose(list(result_edge.start_point()), (0, 0, 0))
        assert np.allclose(list(result_edge.end_point()), (1, 1, 0))

    def test_parse_edge_circle(self, cad_sequence):
        center = Vertex(x=0, y=0)
        radius = 1.0
        circle = Circle(center=center, radius=radius)

        # Parse the edge and verify geometry
        result_edge = cad_sequence._parse_edge(circle)
        assert isinstance(result_edge, Build123dEdge)
        assert np.allclose(list(result_edge.arc_center), (0, 0, 0))
        assert math.isclose(result_edge.radius, radius, rel_tol=1e-6)

    def test_parse_edge_arc(self, cad_sequence):
        center = Vertex(x=0, y=0)
        radius = 1.0
        start_point = Vertex(x=1, y=0)  # Start at (1, 0)
        end_point = Vertex(x=0, y=1)  # End at (0, 1)
        arc = Arc(
            center=center,
            radius=radius,
            start_angle=0,
            end_angle=math.pi / 2,  # Quarter circle
            start_point=start_point,
            end_point=end_point,
        )

        # Parse the edge and verify geometry
        result_edge = cad_sequence._parse_edge(arc)
        assert isinstance(result_edge, Build123dEdge)
        assert np.allclose(list(result_edge.start_point()), (1, 0, 0))
        assert np.allclose(list(result_edge.end_point()), (0, 1, 0), atol=1e-3)
        # Check arc radius
        assert math.isclose(result_edge.radius, radius, rel_tol=1e-5)


@pytest.mark.parametrize("file", ["0000/00007848.json", "0002/00029724.json"])
def test_normalization(file, debug=True):
    """
    Test if the normalization of the CAD sequence is correct (unit cube)
    :param file:
    :return:
    """
    json_file_path = os.path.join(DATA_FOLDER, file)

    # Parse and process the CAD file
    skex_seq: Cad = DeepCadJsonParser.process(json_file_path)
    skex_seq.apply_data_cleaning(visualize_steps=False)
    extrude_before = skex_seq.construction_history[0].extrude.extent_one
    if debug:
        visualize_lines(skex_seq.construction_history[0].sketch[0].outer_wire.edges)
        plt.title("Original CAD Sequence")
        plt.show()
    skex_seq.normalize()

    # Check normalization and visualize lines and centers
    sum_center = np.zeros(2)
    for skex in skex_seq.construction_history:
        sketch = skex.sketch
        assert (
            sketch.bbox_size <= 1
        ), f"Sketch bbox_size is not normalized for file {file}"

        # Calculate bounding box center
        bbox_max = sketch.bbox[0]
        bbox_min = sketch.bbox[1]
        bbox_center = (bbox_max + bbox_min) / 2
        sum_center += bbox_center

        if debug:
            ax = visualize_lines(sketch.outer_wire.edges)
            ax.plot(
                bbox_center[0], bbox_center[1], "ro"
            )  # 'ro' plots red dots for each center
            plt.title("Normalized CAD Sequence")
            plt.show()

    assert np.allclose(sum_center / len(skex_seq.construction_history), np.zeros(2))
    assert extrude_before != skex_seq.construction_history[0].extrude.extent_one


def test_numericalization():
    cad = cad_from_id("00007848")
    cad.apply_data_cleaning(visualize_steps=False)
    cad.normalize()
    cad.render_3d_interactive(title="Original CAD Sequence")
    cad.numericalize(n=QUANTIZATION_SCALE)
    cad.denumericalize(n=QUANTIZATION_SCALE)
    cad.render_3d_interactive(title="Denumericalized CAD Sequence")


def test_denumericalization_from_vector():
    """Test the denumericalization as part of the from_vector method"""
    cad = cad_from_id("00007848")
    cad.apply_data_cleaning(visualize_steps=False)
    cad.normalize()
    cad.render_3d_interactive(title="Original CAD Sequence")
    cad.numericalize(n=QUANTIZATION_SCALE)
    vec = cad.to_vector()
    reconstructed_cad = Cad.from_vector(vec, is_numerical=True)
    reconstructed_cad.render_3d_interactive(title="Reconstructed CAD Sequence")
    assert True


@pytest.mark.parametrize("file", ["23231_efe613bb_0018.json"])
def test_json_conversion_1(file):
    """
    Test if export then import of the JSON leads to the same CAD sequence
    :return:
    """
    json_file_path = os.path.join(DATA_FOLDER, file)

    # Parse and process the CAD file
    skex_seq: Cad = Fusion360GalleryParser.parse(json_file_path)
    skex_seq.render_3d(title="Original CAD Sequence")
    json.dump(skex_seq.to_json(), open("skex_seq.json", "w"), indent=4)
    reconstructed_skex_seq = Cad.from_json(json.load(open("skex_seq.json")))
    reconstructed_skex_seq.render_3d(title="Reconstructed CAD Sequence")


@pytest.mark.parametrize("file", ["skex_seq.json"])
def test_json_conversion_2(file):
    """
    Test is the JSON import then  export leads to the same JSON
    :param file:
    :return:
    """
    json_content = open(file, "r").read()
    j = json.loads(json_content)
    reconstructed_json = Cad.from_json(j).to_json()
    assert reconstructed_json == j, "Reconstructed JSON does not match original JSON"


@pytest.mark.parametrize("file", ["23231_efe613bb_0018.json"])
def test_python_conversion(file):
    """
    Test if the python code conversion is correct
    :param file:
    :return:
    """
    json_file_path = os.path.join(DATA_FOLDER, file)

    # Parse and process the CAD file
    skex_seq: Cad = Fusion360GalleryParser.parse(json_file_path)
    code = skex_seq.to_python()
    with open("skex_seq.py", "w") as f:
        f.write(code)


def test_to_vector(two_dimensions_cad_sequence):
    two_dimensions_cad_sequence.normalize()
    two_dimensions_cad_sequence.plot()
    two_dimensions_cad_sequence.numericalize(n=256)
    two_dimensions_cad_sequence.plot("Numericalized CAD Sequence")
    two_dimensions_cad_sequence.denumericalize(n=256)
    two_dimensions_cad_sequence.plot("Denumericalized CAD Sequence")
    vec = two_dimensions_cad_sequence.to_vector()
    assert vec is not None


def test_to_vector_1():
    cad = cad_from_id("0001/00016622")
    cad.apply_data_cleaning(visualize_steps=False)
    cad.normalize()
    cad.numericalize(n=256)
    cad.show_3d()
    cad_reconstructed = Cad.from_vector(cad.to_vector(), is_numerical=True)
    df = pd.DataFrame(cad_reconstructed.to_vector())
    print(df.to_markdown)
    cad_reconstructed.show_3d()
    pass


def test_to_vector_2():
    """Good test case because of internal holes"""
    cad = cad_from_id("00016622")
    cad.render_3d_interactive(title="Original CAD Sequence")
    cad_reconstructed = cad.from_vector(cad.to_vector())
    cad_reconstructed.render_3d_interactive(title="Reconstructed CAD Sequence")
    assert np.array_equal(cad_reconstructed.to_vector(), cad.to_vector())


def test_from_vector():
    """
    Test from vector with AI model output
    :return:
    """
    vector = np.array(
        [
            261,
            140,
            122,
            0,
            0,
            0,
            0,
            255,
            98,
            197,
            260,
            98,
            98,
            260,
            0,
            0,
            255,
            0,
            197,
            260,
            73,
            1,
            0,
            0,
            0,
            0,
            261,
            119,
            130,
            7,
            0,
            0,
            0,
            257,
            244,
            71,
            12,
            260,
            5,
            0,
            0,
            0,
            0,
            258,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]
    )
    vector = np.array(
        [
            262,
            125,
            127,
            0,
            0,
            0,
            0,
            256,
            255,
            255,
            256,
            255,
            0,
            256,
            0,
            0,
            256,
            0,
            255,
            261,
            26,
            1,
            0,
            0,
            0,
            0,
            259,
            259,
            259,
            259,
            0,
            0,
            0,
            259,
            259,
            259,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    cad = Cad.from_vector(vector)
    cad.render_3d_interactive()


@pytest.mark.parametrize("file", ["0000/00007848.json", "0002/00029724.json"])
def test_vector_conversion(file):
    """
    Test if the normalization of the CAD sequence is correct (unit cube)
    :param file:
    :return:
    """
    json_file_path = os.path.join(DATA_FOLDER, file)

    # Parse and process the CAD file
    cad: Cad = DeepCadJsonParser.process(json_file_path)
    cad.apply_data_cleaning(visualize_steps=False)
    # before visualization
    cad.render_3d("Before")
    print("Before")
    print(pretty_print_cad_vector(cad.to_vector()))
    cad = Cad.from_vector(cad.to_vector())
    print("After")
    print(pretty_print_cad_vector(cad.to_vector()))
    # after visualization
    cad.render_3d("After")


def test_graph_format(flash_cad):
    """Good test case because of internal holes"""
    flash_cad.show_3d()
    graph_format = flash_cad.to_graph_format()
    cad_reconstructed = Cad.from_graph_format(graph_format)
    cad_reconstructed.show_3d()


def test_from_json():
    import json

    from rapidcadpy.cad import Cad

    json = json.loads(
        open("/Users/elias.berger/cadgpt/server/mock_data/demo_2.stp.json").read()
    )
    cad = Cad.from_json(json)
    cad.to_step("/Users/elias.berger/cadgpt/server/mock_data/demo_2.stp")


def test_to_python(long_sequence_omni):
    long_sequence_omni.normalize()
    python_str = long_sequence_omni.to_python()
    reconstructed_cad = execute_cad_code(python_str)
    reconstructed_cad.show_3d()


def test_from_python(random_py_file):
    reconstructed_cad = execute_cad_code(random_py_file)
    reconstructed_cad.show_3d()
