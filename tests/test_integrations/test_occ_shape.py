"""
Unit tests for OCC Shape export methods.
"""

import os

from rapidcadpy.integrations.occ import OpenCascadeApp
from rapidcadpy.integrations.occ.shape import OccShape


class TestOccShapeExport:
    """Test cases for OCC Shape export methods."""

    def test_to_step_simple_cube(self):
        """Test that to_step exports a simple cube to a STEP file."""
        # Create a simple cube
        app = OpenCascadeApp()
        wp = app.work_plane("XY")
        cube = (
            wp.move_to(0, 0)
            .line_to(10, 0)
            .line_to(10, 10)
            .line_to(0, 10)
            .line_to(0, 0)
            .extrude(10)
        )

        test_file = "test_cube_export.step"

        try:
            # Export to STEP
            cube.to_step(test_file)

            # Verify file was created
            assert os.path.exists(test_file), "STEP file should be created"

            # Verify file has content
            file_size = os.path.getsize(test_file)
            assert file_size > 0, "STEP file should have content"

            # Basic validation - STEP files should start with "ISO-10303-21;"
            with open(test_file, "r") as f:
                first_line = f.readline().strip()
                assert (
                    first_line == "ISO-10303-21;"
                ), "STEP file should start with ISO-10303-21;"

        finally:
            # Clean up
            if os.path.exists(test_file):
                os.remove(test_file)


def test_occ_shape_implements_edge_contract_and_fillet():
    """The OCC shape backend must remain concrete after Shape API additions."""

    app = OpenCascadeApp()
    box = app.work_plane("XY").rect(3.0, 3.0).close().extrude(0.5)

    assert isinstance(box, OccShape)
    assert len(box._raw_edges()) == 12
    original_volume = box.volume()

    result = box.edges("|Z").fillet(0.125)

    assert result is box
    assert 0.0 < box.volume() < original_volume


def test_documented_occ_import_and_export_alias(tmp_path):
    """The Quick Start example should work without private module imports."""

    app = OpenCascadeApp()
    app.new_document()
    cube = app.work_plane("XY").move_to(-5, -5).rect(10, 10).extrude(10)
    step_path = tmp_path / "my_cube.step"

    cube.export(str(step_path))

    assert step_path.read_text(encoding="utf-8").startswith("ISO-10303-21;")
