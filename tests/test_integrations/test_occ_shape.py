"""
Unit tests for OCC Shape export methods.
"""

import os

import pytest

from rapidcadpy.integrations.occ.app import OpenCascadeApp


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
