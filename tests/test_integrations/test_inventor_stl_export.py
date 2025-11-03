"""
Unit tests for STL export functionality with Inventor backend.

This module tests the to_stl() method of the InventorApp class,
including exporting both newly created geometries and existing .ipt files.
"""

import os
import pathlib
import tempfile

import pytest

from rapidcadpy.integrations.inventor.app import InventorApp


@pytest.fixture
def inventor_app():
    """Fixture to provide a clean Inventor application instance."""
    app = InventorApp()
    app.new_document()
    yield app


@pytest.fixture
def sample_ipt_drop_arm():
    """Fixture to provide path to the drop_arm.ipt test file."""
    path = pathlib.Path(__file__).parent.parent / "test_files" / "drop_arm.ipt"
    return path.resolve()


class TestInventorSTLExport:
    """Test cases for STL export with Inventor backend."""

    @pytest.mark.skipif(
        not pytest.importorskip("win32com", reason="win32com not available"),
        reason="win32com not available",
    )
    def test_export_simple_cube_to_stl(self, inventor_app):
        """Test creating a simple cube and exporting to STL."""
        # Create a simple cube using fluent API
        workplane = inventor_app.work_plane("XY")
        cube = workplane.rect(10, 10).extrude(10)

        # Export to STL in temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            stl_file = os.path.join(temp_dir, "test_cube.stl")
            inventor_app.to_stl(stl_file)

            # Verify the file was created
            assert os.path.exists(stl_file), "STL file was not created"

            # Verify file has content
            assert os.path.getsize(stl_file) > 0, "STL file is empty"

    @pytest.mark.skipif(
        not pytest.importorskip("win32com", reason="win32com not available"),
        reason="win32com not available",
    )
    def test_export_cylinder_to_stl(self, inventor_app):
        """Test creating a cylinder and exporting to STL."""
        # Create a cylinder
        workplane = inventor_app.work_plane("XY")
        cylinder = workplane.circle(5).extrude(20)

        # Export to STL
        with tempfile.TemporaryDirectory() as temp_dir:
            stl_file = os.path.join(temp_dir, "test_cylinder.stl")
            inventor_app.to_stl(stl_file)

            # Verify the file was created and has content
            assert os.path.exists(stl_file), "STL file was not created"
            assert os.path.getsize(stl_file) > 0, "STL file is empty"

    @pytest.mark.skipif(
        not pytest.importorskip("win32com", reason="win32com not available"),
        reason="win32com not available",
    )
    def test_export_with_auto_stl_extension(self, inventor_app):
        """Test that .stl extension is automatically added if not provided."""
        # Create a simple shape
        workplane = inventor_app.work_plane("XY")
        cube = workplane.rect(5, 5).extrude(5)

        # Export without .stl extension
        with tempfile.TemporaryDirectory() as temp_dir:
            base_name = os.path.join(temp_dir, "test_cube_no_ext")
            inventor_app.to_stl(base_name)

            # Verify file was created with .stl extension
            expected_file = base_name + ".stl"
            assert os.path.exists(
                expected_file
            ), "STL file with auto extension was not created"
            assert os.path.getsize(expected_file) > 0, "STL file is empty"

    @pytest.mark.skipif(
        not pytest.importorskip("win32com", reason="win32com not available"),
        reason="win32com not available",
    )
    def test_export_from_existing_ipt_file(self, sample_ipt_drop_arm):
        """Test exporting an existing .ipt file to STL."""
        # Skip if test file doesn't exist
        if not os.path.exists(sample_ipt_drop_arm):
            pytest.skip(f"Test file not found: {sample_ipt_drop_arm}")

        # Open existing IPT file
        app = InventorApp()
        app.open_document(str(sample_ipt_drop_arm))

        # Export to STL
        with tempfile.TemporaryDirectory() as temp_dir:
            stl_file = os.path.join(temp_dir, "drop_arm_export.stl")
            app.to_stl(stl_file)

            # Verify the file was created and has content
            assert os.path.exists(
                stl_file
            ), "STL file was not created from existing IPT"
            assert os.path.getsize(stl_file) > 0, "STL file is empty"
