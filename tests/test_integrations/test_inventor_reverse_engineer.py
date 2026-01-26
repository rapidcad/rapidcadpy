import os
import pathlib
import tempfile

import pytest

from rapidcadpy.integrations.inventor.app import InventorApp
from rapidcadpy.integrations.inventor.reverse_engineer import InventorReverseEngineer
from rapidcadpy.integrations.inventor.workplane import InventorWorkPlane


@pytest.fixture
def inventor_app():
    """Fixture to provide a clean Inventor application instance."""
    app = InventorApp()
    yield app


@pytest.fixture
def sample_ipt_file(inventor_app):
    """Fixture to create a sample IPT file for testing."""
    # Create a simple geometry
    wp = InventorWorkPlane(app=inventor_app)
    wp.move_to(0, 0).line_to(10, 0).line_to(10, 10).line_to(0, 10).line_to(0, 0)
    wp.extrude(5.0)

    # Save to temporary file
    temp_dir = tempfile.mkdtemp()
    ipt_file = os.path.join(temp_dir, "test_part.ipt")
    inventor_app.app.ActiveDocument.SaveAs(ipt_file, False)

    yield ipt_file

    # Cleanup
    if os.path.exists(ipt_file):
        os.remove(ipt_file)
    os.rmdir(temp_dir)


@pytest.fixture
def sample_ipt_cube():
    path = (
        pathlib.Path(__file__).parent.parent / "test_files" / "test_cube_10x10x10.ipt"
    )
    return path.resolve()


@pytest.fixture
def sample_ipt_simple_shaft():
    path = pathlib.Path(__file__).parent.parent / "test_files" / "simple_shaft.ipt"
    return path.resolve()


@pytest.fixture
def sample_ipt_complex_shaft():
    path = pathlib.Path(__file__).parent.parent / "test_files" / "generated_shaft.ipt"
    return path.resolve()


@pytest.fixture
def sample_ipt_complex_shaft_6():
    path = pathlib.Path(__file__).parent.parent / "test_files" / "5.ipt"
    return path.resolve()


@pytest.fixture
def sample_ipt_offset_sketch_plane():
    path = (
        pathlib.Path(__file__).parent.parent / "test_files" / "offset_sketch_plane.ipt"
    )
    return path.resolve()


def reverse_engineer_ipt(file_path: str, output_file: str = None) -> str:
    """
    Reverse engineer an IPT file to Python code.

    Args:
        file_path: Path to the IPT file
        output_file: Optional path to save the generated code

    Returns:
        Generated Python code as string
    """
    # Initialize Inventor
    app = InventorApp(headless=True)
    doc = app.open_document(file_path)

    # Create reverse engineer instance
    engineer = InventorReverseEngineer(doc)

    # Generate code
    generated_code = engineer.analyze_ipt_file()

    # Save to file if specified
    if output_file:
        with open(output_file, "w") as f:
            f.write(generated_code)
        print(f"Generated code saved to: {output_file}")

    return generated_code


class TestInventorReverseEngineer:
    """Test class for Inventor reverse engineering functionality."""

    def test_reverse_engineer_basic_extrusion(self, sample_ipt_cube):
        """Test reverse engineering a basic extruded rectangle."""
        print(sample_ipt_cube)
        assert os.path.exists(sample_ipt_cube)
        generated_code = reverse_engineer_ipt(sample_ipt_cube)

        # Verify the generated code contains expected elements
        print(generated_code)
        assert (
            "from rapidcadpy.integrations.inventor import InventorApp" in generated_code
        )
        assert "app = InventorApp()" in generated_code
        assert "line_to" in generated_code
        assert "extrude" in generated_code

    def test_reverse_engineer_simple_shaft(self, sample_ipt_simple_shaft):
        """Test reverse engineering a simple shaft."""
        print(sample_ipt_simple_shaft)
        assert os.path.exists(sample_ipt_simple_shaft)
        generated_code = reverse_engineer_ipt(sample_ipt_simple_shaft)

        # Verify the generated code contains expected elements
        print(generated_code)
        assert (
            "from rapidcadpy.integrations.inventor import InventorApp" in generated_code
        )
        assert "app = InventorApp()" in generated_code
        assert "line_to" in generated_code
        assert "revolve" in generated_code

    def test_reverse_engineer_offset_sketch_plane(self, sample_ipt_offset_sketch_plane):
        """Test reverse engineering a complex shaft."""
        print(sample_ipt_offset_sketch_plane)
        assert os.path.exists(sample_ipt_offset_sketch_plane)
        generated_code = reverse_engineer_ipt(sample_ipt_offset_sketch_plane)

        # Verify the generated code contains expected elements
        print(generated_code)
        assert "from rapidcadpy import InventorApp" in generated_code
        assert "app = InventorApp()" in generated_code
        assert "line_to" in generated_code
        assert "revolve" in generated_code

    def test_reverse_engineer_complex_shaft(self, sample_ipt_complex_shaft):
        """Test reverse engineering a complex shaft."""
        print(sample_ipt_complex_shaft)
        assert os.path.exists(sample_ipt_complex_shaft)
        generated_code = reverse_engineer_ipt(sample_ipt_complex_shaft)

        # Verify the generated code contains expected elements
        print(generated_code)
        assert "from rapidcadpy import InventorApp" in generated_code
        assert "app = InventorApp()" in generated_code
        assert "line_to" in generated_code
        assert "revolve" in generated_code

    def test_reverse_engineer_complex_shaft_2(self, sample_ipt_complex_shaft_6):
        """Test reverse engineering a complex shaft."""
        print(sample_ipt_complex_shaft_6)
        assert os.path.exists(sample_ipt_complex_shaft_6)
        generated_code = reverse_engineer_ipt(sample_ipt_complex_shaft_6)

        # Verify the generated code contains expected elements
        print(generated_code)
        assert "from rapidcadpy import InventorApp" in generated_code
        assert "app = InventorApp()" in generated_code
        assert "line_to" in generated_code
        assert "revolve" in generated_code
