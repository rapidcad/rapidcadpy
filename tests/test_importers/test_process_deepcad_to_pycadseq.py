import pathlib

import pytest

from rapidcadpy.json_importer.process_deepcad_to_pycadseq import (
    DeepCadToPyCadSeqGenerator,
    generate_pycadseq_code_from_deepcad_json,
)


@pytest.fixture
def sample_deepcad_json():
    """Fixture to provide path to sample DeepCAD JSON file."""
    path = pathlib.Path(__file__).parent / "test_files" / "00000016.json"
    print(path)
    if not path.exists():
        pytest.skip(f"Test file not found: {path}")
    return str(path.resolve())


@pytest.fixture
def sample_deepcad_json_2():
    """Fixture to provide path to sample DeepCAD JSON file."""
    path = pathlib.Path(__file__).parent / "test_files" / "00561062.json"
    print(path)
    if not path.exists():
        pytest.skip(f"Test file not found: {path}")
    return str(path.resolve())


@pytest.fixture
def sample_deepcad_json_3():
    """Fixture to provide path to sample DeepCAD JSON file."""
    path = pathlib.Path(__file__).parent / "test_files" / "00000767.json"
    if not path.exists():
        pytest.skip(f"Test file not found: {path}")
    return str(path.resolve())


@pytest.fixture
def sample_deepcad_json_shield():
    """Fixture to provide path to sample DeepCAD JSON file."""
    path = pathlib.Path(__file__).parent / "test_files" / "00950156.json"
    if not path.exists():
        pytest.skip(f"Test file not found: {path}")
    return str(path.resolve())


@pytest.fixture
def sample_deepcad_json_flash():
    """Fixture to provide path to sample DeepCAD JSON file."""
    path = pathlib.Path(__file__).parent / "test_files" / "00561068_flash.json"
    if not path.exists():
        pytest.skip(f"Test file not found: {path}")
    return str(path.resolve())

@pytest.fixture
def sample_deepcad_json_internal_holes():
    """Fixture to provide path to sample DeepCAD JSON file."""
    path = pathlib.Path(__file__).parent / "test_files" / "00007848_internal_holes.json"
    if not path.exists():
        pytest.skip(f"Test file not found: {path}")
    return str(path.resolve())

class TestDeepCadToPyCadSeqGenerator:
    """Test class for DeepCAD to PyCadSeq code generation."""

    def test_generator_init_inventor(self):
        """Test generator initialization with Inventor backend."""
        generator = DeepCadToPyCadSeqGenerator(backend="inventor")
        assert generator.backend == "inventor"
        assert generator.generated_code == []

    def test_generator_init_occ(self):
        """Test generator initialization with OCC backend."""
        generator = DeepCadToPyCadSeqGenerator(backend="occ")
        assert generator.backend == "occ"
        assert generator.generated_code == []

    def test_generator_init_invalid_backend(self):
        """Test generator initialization with invalid backend."""
        generator = DeepCadToPyCadSeqGenerator(backend="invalid")
        assert generator.backend == "invalid"

        # Should raise error when generating code
        with pytest.raises(ValueError, match="Unsupported backend"):
            generator.generate_code_from_json("dummy.json")

    def test_generate_code_inventor_backend(self, sample_deepcad_json):
        """Test code generation with Inventor backend."""
        generated_code = generate_pycadseq_code_from_deepcad_json(
            sample_deepcad_json, backend="inventor"
        )

        print(generated_code)

        # Verify the generated code contains expected elements
        assert "from rapidcadpy import InventorApp" in generated_code
        assert "app = InventorApp()" in generated_code
        assert "app.new_document()" in generated_code
        assert 'app.work_plane("XY")' in generated_code
        assert "move_to" in generated_code
        assert "line_to" in generated_code
        assert "three_point_arc" in generated_code
        assert "extrude" in generated_code

        # Should contain sketch and extrude comments
        assert "# Sketch 1" in generated_code
        assert "# Extrude feature 1" in generated_code

    def test_generate_code_inventor_backend_2(self, sample_deepcad_json_2):
        """Test code generation with Inventor backend."""
        generated_code = generate_pycadseq_code_from_deepcad_json(
            sample_deepcad_json_2, backend="inventor"
        )

        print(generated_code)

        # Verify the generated code contains expected elements
        assert "from rapidcadpy import InventorApp" in generated_code
        assert "app = InventorApp()" in generated_code
        assert "app.new_document()" in generated_code
        assert 'app.work_plane("XY")' in generated_code
        assert "move_to" in generated_code
        assert "line_to" in generated_code
        assert "three_point_arc" in generated_code
        assert "extrude" in generated_code

        # Should contain sketch and extrude comments
        assert "# Sketch 1" in generated_code
        assert "# Extrude feature 1" in generated_code

    def test_generate_code_inventor_backend_3(self, sample_deepcad_json_3):
        generated_code = generate_pycadseq_code_from_deepcad_json(
            sample_deepcad_json_3, backend="inventor"
        )

        print(generated_code)
        assert (
            "shape1 = wp1.extrude(0.0889, 'NewBodyFeatureOperation')" in generated_code
        )

    def test_generate_code_inventor_backend_4(self, sample_deepcad_json_shield):
        generated_code = generate_pycadseq_code_from_deepcad_json(
            sample_deepcad_json_shield, backend="inventor"
        )

        print(generated_code)

    def test_generate_code_inventor_backend_flash(self, sample_deepcad_json_flash):
        generated_code = generate_pycadseq_code_from_deepcad_json(
            sample_deepcad_json_flash, backend="inventor"
        )

        print(generated_code)

    def test_generate_code_internal_holes(self, sample_deepcad_json_internal_holes):
        generated_code = generate_pycadseq_code_from_deepcad_json(
            sample_deepcad_json_internal_holes, backend="inventor"
        )

        print(generated_code)