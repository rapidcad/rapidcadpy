import pytest
import pathlib
import os
from rapidcadpy.json_importer.process_deepcad_to_pycadseq import (
    DeepCadToPyCadSeqGenerator,
    generate_pycadseq_code_from_deepcad_json
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
    print(path)
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
            sample_deepcad_json, 
            backend="inventor"
        )

        print(generated_code)
        
        # Verify the generated code contains expected elements
        assert "from rapidcadpy import InventorApp" in generated_code
        assert "app = InventorApp()" in generated_code
        assert "app.new_document()" in generated_code
        assert "app.work_plane(\"XY\")" in generated_code
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
            sample_deepcad_json_2, 
            backend="inventor"
        )

        print(generated_code)
        
        # Verify the generated code contains expected elements
        assert "from rapidcadpy import InventorApp" in generated_code
        assert "app = InventorApp()" in generated_code
        assert "app.new_document()" in generated_code
        assert "app.work_plane(\"XY\")" in generated_code
        assert "move_to" in generated_code
        assert "line_to" in generated_code
        assert "three_point_arc" in generated_code
        assert "extrude" in generated_code
        
        # Should contain sketch and extrude comments
        assert "# Sketch 1" in generated_code
        assert "# Extrude feature 1" in generated_code

    def test_generate_code_inventor_backend_3(self, sample_deepcad_json_3):
        generated_code = generate_pycadseq_code_from_deepcad_json(
            sample_deepcad_json_3, 
            backend="inventor"
        )

        print(generated_code)
        assert "shape1 = wp1.extrude(0.0889, 'NewBodyFeatureOperation')" in generated_code
        
    
    def test_generate_code_occ_backend(self, sample_deepcad_json):
        """Test code generation with OCC backend."""
        generated_code = generate_pycadseq_code_from_deepcad_json(
            sample_deepcad_json, 
            backend="occ"
        )
        
        # Verify the generated code contains expected elements
        assert "from pycadseq import OpenCascadeApp" in generated_code
        assert "app = OpenCascadeApp()" in generated_code
        assert "app.work_plane(\"XY\")" in generated_code
        assert "move_to" in generated_code
        assert "line_to" in generated_code
        assert "three_point_arc" in generated_code
        assert "extrude" in generated_code
        
        # Should not contain new_document() for OCC
        assert "app.new_document()" not in generated_code
        
        print("Generated OCC code:")
        print(generated_code)
    
    def test_workplane_detection_xy_plane(self):
        from pycadseq import InventorApp

        # Initialize Inventor application
        app = InventorApp()
        app.new_document()

        # Sketch 1
        wp1 = app.work_plane("XY")

        wp1.move_to(0.254, 0.0).line_to(0.0254, 0.0).three_point_arc((0.017961, 0.017961), (0.0, -0.0254)).line_to(0.0, -0.0254).line_to(0.254, -0.254).line_to(0.254, -0.254)
        # Extrude feature 1
        shape1 = wp1.extrude(0.012700000000000001, 'NewBodyFeatureOperation')
    
    def test_geometry_extraction(self, sample_deepcad_json):
        """Test that geometry is correctly extracted from JSON."""
        generated_code = generate_pycadseq_code_from_deepcad_json(sample_deepcad_json)
        
        # Based on the JSON file, should contain specific coordinates
        # Line from (0.254, 0.0) to (0.0254, 0.0)
        assert "move_to(0.254, 0.0)" in generated_code
        assert "line_to(0.0254, 0.0)" in generated_code
        
        # Arc with center at (0.0, 0.0) and radius 0.0254
        assert "three_point_arc" in generated_code
        
        # Extrude with distance 0.0127
        assert "extrude(0.012700000000000001" in generated_code or "extrude(0.0127" in generated_code
    
    def test_multiple_profiles_handling(self, sample_deepcad_json):
        """Test handling of sketches with multiple profiles."""
        generator = DeepCadToPyCadSeqGenerator(backend="inventor")
        generated_code = generator.generate_code_from_json(sample_deepcad_json)
        
        # The JSON has multiple profiles (JGC, JGK, JGG) but only JGG is used in extrude
        # Should generate code for the profiles
        assert "wp1" in generated_code
    
    def test_extrude_operation_mapping(self, sample_deepcad_json):
        """Test that extrude operations are correctly mapped."""
        generated_code = generate_pycadseq_code_from_deepcad_json(sample_deepcad_json)
        
        # Should map NewBodyFeatureOperation correctly
        assert "NewBodyFeatureOperation" in generated_code
    
    def test_file_output(self, sample_deepcad_json, tmp_path):
        """Test saving generated code to file."""
        output_file = tmp_path / "generated_code.py"
        
        generated_code = generate_pycadseq_code_from_deepcad_json(
            sample_deepcad_json,
            backend="inventor",
            output_file=str(output_file)
        )
        
        # Verify file was created
        assert output_file.exists()
        
        # Verify file content matches returned code
        with open(output_file, 'r') as f:
            file_content = f.read()
        
        assert file_content == generated_code
        assert len(file_content) > 0
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent JSON file."""
        with pytest.raises(FileNotFoundError):
            generate_pycadseq_code_from_deepcad_json("nonexistent.json")
    
    def test_code_structure_inventor(self, sample_deepcad_json):
        """Test that generated code has proper structure for Inventor."""
        generated_code = generate_pycadseq_code_from_deepcad_json(
            sample_deepcad_json, 
            backend="inventor"
        )
        lines = generated_code.split('\n')
        
        # Check for proper imports at the beginning
        import_lines = [line for line in lines[:5] if line.startswith("from")]
        assert len(import_lines) >= 1
        assert "InventorApp" in import_lines[0]
        
        # Check for app initialization
        app_init_lines = [line for line in lines if "app = InventorApp()" in line]
        assert len(app_init_lines) == 1
        
        # Check for new_document call
        new_doc_lines = [line for line in lines if "app.new_document()" in line]
        assert len(new_doc_lines) == 1
        
        # Check for workplane creation
        workplane_lines = [line for line in lines if "work_plane(" in line]
        assert len(workplane_lines) >= 1
    
    def test_code_structure_occ(self, sample_deepcad_json):
        """Test that generated code has proper structure for OCC."""
        generated_code = generate_pycadseq_code_from_deepcad_json(
            sample_deepcad_json, 
            backend="occ"
        )
        lines = generated_code.split('\n')
        
        # Check for proper imports at the beginning
        import_lines = [line for line in lines[:5] if line.startswith("from")]
        assert len(import_lines) >= 1
        assert "OCCApp" in import_lines[0]
        
        # Check for app initialization
        app_init_lines = [line for line in lines if "app = OCCApp()" in line]
        assert len(app_init_lines) == 1
        
        # Should NOT have new_document for OCC
        new_doc_lines = [line for line in lines if "app.new_document()" in line]
        assert len(new_doc_lines) == 0


@pytest.mark.parametrize("backend", ["inventor", "occ"])
def test_backends_parametrized(sample_deepcad_json, backend):
    """Parametrized test for both backends."""
    generated_code = generate_pycadseq_code_from_deepcad_json(
        sample_deepcad_json,
        backend=backend
    )
    
    # Common assertions for both backends
    assert f"from pycadseq.integrations.{backend}" in generated_code
    assert f"app = {backend.upper() if backend == 'occ' else backend.title()}App()" in generated_code
    assert "work_plane" in generated_code
    assert "extrude" in generated_code
    assert len(generated_code) > 100  # Should generate substantial code


if __name__ == "__main__":
    # Run pytest programmatically
    pytest.main([__file__, "-v"])
