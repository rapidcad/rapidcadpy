import pathlib
import pytest
import os
import tempfile
from pathlib import Path
from rapidcadpy.integrations.inventor.app import InventorApp
from rapidcadpy.integrations.inventor.workplane import InventorWorkPlane
from rapidcadpy.integrations.inventor.reverse_engineer import InventorReverseEngineer


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
    path = pathlib.Path(__file__).parent.parent / "test_files" / "test_cube_10x10x10.ipt"
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
def sample_ipt_complex_shaft_2():
    path = pathlib.Path(__file__).parent.parent / "test_files" / "generated_shaft_2.ipt"
    return path.resolve()

@pytest.fixture
def sample_ipt_offset_sketch_plane():
    path = pathlib.Path(__file__).parent.parent / "test_files" / "offset_sketch_plane.ipt"
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
        with open(output_file, 'w') as f:
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
        assert "from rapidcadpy.integrations.inventor import InventorApp" in generated_code
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
        assert "from rapidcadpy.integrations.inventor import InventorApp" in generated_code
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

    def test_reverse_engineer_complex_shaft_2(self, sample_ipt_complex_shaft):
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
    
    def test_reverse_engineer_results(self):
        from rapidcadpy import InventorApp

        # Initialize Inventor application
        app = InventorApp(headless=False)
        app.new_document()

        # Sketch 1
        wp1 = app.work_plane("YZ")

        wp1.move_to(0.0, 1.770599).line_to(0.318508, 1.770599).line_to(0.318508, 0.532619).line_to(0.527386, 0.532619).line_to(0.527386, 0.259112).line_to(0.389785, 0.259112).line_to(0.389785, -0.021593).line_to(0.528091, -0.021593).line_to(0.528091, -0.332441).line_to(0.389785, -0.330076).line_to(0.382138, -0.777418).line_to(0.0, -0.777418).line_to(0.0, 1.770599)

        # Revolve feature 1
        shape1 = wp1.revolve(6.283185307179586, (0.0, 0.0), 'NewBodyFeatureOperation')

    def test_reverse_engineer_results_2(self):
        from rapidcadpy import InventorApp

        # Initialize Inventor application
        app = InventorApp(headless=False)
        app.new_document()

        # Sketch 1
        wp1 = app.work_plane("XY")

        wp1.move_to(0.0, 0.0).line_to(0.0, 2.1).line_to(10.7, 2.1).line_to(10.7, 2.75).line_to(12.3, 2.75).line_to(12.3, 3.25).line_to(14.815, 3.25).line_to(14.815, 4.0).line_to(16.915, 4.0).line_to(16.915, 3.2).line_to(34.115, 3.2).line_to(34.115, 2.5).line_to(36.38, 2.5).line_to(36.38, 0.0).line_to(0.0, 0.0)

        # Revolve feature 1
        shape1 = wp1.revolve(6.283185307179586, (0.0, 0.0), 'NewBodyFeatureOperation')

        # Sketch 2
        wp2 = app.work_plane("XY")

        wp2.move_to(0.85, 0.0).line_to(9.85, 0.0)
        wp2.move_to(1.45, -0.6).line_to(9.25, -0.6).three_point_arc((9.85, 0.0), (9.25, 0.6)).line_to(1.45, 0.6).three_point_arc((0.85, 0.0), (1.45, -0.6))

        # Extrude feature 2
        shape2 = wp2.extrude(10.0, 'Cut')

        # Sketch 3
        wp3 = app.work_plane("XY")

        wp3.move_to(10.7, 2.3).line_to(10.72, 2.225359)
        wp3.move_to(10.561962, 2.07).line_to(10.45, 2.1).line_to(10.45, 2.3).line_to(10.7, 2.3)
        wp3.move_to(10.561962, 2.07).three_point_arc((10.695525, 2.092195), (10.72, 2.225359))

        # Revolve feature 3
        shape3 = wp3.revolve(6.283185307179586, (0.0, 0.0), 'Cut')

        # Sketch 4
        wp4 = app.work_plane("XY")

        wp4.move_to(0.0, 0.0).line_to(0.0, 0.5).line_to(0.095263, 0.335).line_to(0.402702, 0.1575).line_to(0.609067, 0.1575).line_to(0.7, 0.0).line_to(0.0, 0.0)

        # Revolve feature 4
        shape4 = wp4.revolve(6.283185307179586, (0.0, 0.0), 'Cut')

        # Sketch 5
        wp5 = app.work_plane("XY")

        wp5.move_to(12.3, 2.95).line_to(12.32, 2.875359)
        wp5.move_to(12.161962, 2.72).line_to(12.05, 2.75).line_to(12.05, 2.95).line_to(12.3, 2.95)
        wp5.move_to(12.161962, 2.72).three_point_arc((12.295525, 2.742195), (12.32, 2.875359))

        # Revolve feature 5
        shape5 = wp5.revolve(6.283185307179586, (0.0, 0.0), 'Cut')

        # Sketch 6
        wp6 = app.work_plane("XY")

        wp6.move_to(12.75, 3.1).line_to(13.015, 3.1).line_to(13.015, 3.25).line_to(12.75, 3.25).line_to(12.75, 3.1)

        # Revolve feature 6
        shape6 = wp6.revolve(6.283185307179586, (0.0, 0.0), 'Cut')

        # Sketch 7
        wp7 = app.work_plane("XY")

        wp7.move_to(14.815, 3.45).line_to(14.835, 3.375359)
        wp7.move_to(14.676962, 3.22).line_to(14.565, 3.25).line_to(14.565, 3.45).line_to(14.815, 3.45)
        wp7.move_to(14.676962, 3.22).three_point_arc((14.810525, 3.242195), (14.835, 3.375359))

        # Revolve feature 7
        shape7 = wp7.revolve(6.283185307179586, (0.0, 0.0), 'Cut')

        # Sketch 8
        wp8 = app.work_plane("XY")

        wp8.move_to(18.515, 0.0).line_to(32.515, 0.0)
        wp8.move_to(19.415, -0.9).line_to(31.615, -0.9).three_point_arc((32.515, 0.0), (31.615, 0.9)).line_to(19.415, 0.9).three_point_arc((18.515, 0.0), (19.415, -0.9))

        # Extrude feature 8
        shape8 = wp8.extrude(10.0, 'Cut')

        # Sketch 9
        wp9 = app.work_plane("XY")

        wp9.move_to(16.915, 3.4).line_to(16.895, 3.325359).three_point_arc((16.919475, 3.192195), (17.053038, 3.17)).line_to(17.165, 3.2).line_to(17.165, 3.4).line_to(16.915, 3.4)

        # Revolve feature 9
        shape9 = wp9.revolve(6.283185307179586, (0.0, 0.0), 'Cut')

        # Sketch 10
        wp10 = app.work_plane("XY")

        wp10.move_to(35.93, 2.35).line_to(35.715, 2.35).line_to(35.715, 2.5).line_to(35.93, 2.5).line_to(35.93, 2.35)

        # Revolve feature 10
        shape10 = wp10.revolve(6.283185307179586, (0.0, 0.0), 'Cut')

        # Sketch 11
        wp11 = app.work_plane("XY")

        wp11.move_to(34.115, 2.7).line_to(34.095, 2.625359).three_point_arc((34.119475, 2.492195), (34.253038, 2.47)).line_to(34.365, 2.5).line_to(34.365, 2.7).line_to(34.115, 2.7)

        # Revolve feature 11
        shape11 = wp11.revolve(6.283185307179586, (0.0, 0.0), 'Cut')