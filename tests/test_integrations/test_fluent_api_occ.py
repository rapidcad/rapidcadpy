"""
Unit tests for the fluent API with OCC backend.

This module tests the fluent API integration with the OCC backend,
including creating simple geometries and exporting to STEP format.
"""

import pytest

from rapidcadpy.integrations.occ.app import OpenCascadeApp


class TestFluentAPIOCCBackend:
    """Test cases for the fluent API with OCC backend."""

    @pytest.mark.skipif(
        not pytest.importorskip("OCP", reason="OCP not available"),
        reason="OCP not available",
    )
    def test_simple_cube_creation_and_export(self):
        # Create a simple cube using fluent API
        app = OpenCascadeApp()
        workplane = app.work_plane("XY")
        cube = (
            workplane.line_to(10, 0)
            .line_to(10, 10)
            .line_to(0, 10)
            .line_to(0, 0)
            .extrude(10)
        )

    def test_simple_cylinder(self):
        # Create a simple cylinder using fluent API
        app = OpenCascadeApp()
        workplane = app.work_plane("XY")
        cylinder = workplane.circle(5).extrude(20)

    def test_three_point_arc_simple(self):
        """Test that three_point_arc creates a valid arc and can be extruded."""
        app = OpenCascadeApp()
        wp = app.work_plane("XY")

        # Create a simple shape with a three-point arc
        # Starting at (0, 0), arc through (5, 5) to (10, 0), then close with lines
        shape = (
            wp.move_to(0, 0)
            .three_point_arc((5, 5), (10, 0))  # Arc from (0,0) through (5,5) to (10,0)
            .line_to(10, -5)  # Line down
            .line_to(0, -5)  # Line left
            .line_to(0, 0)  # Line back to start
            .extrude(10)  # Extrude to create 3D shape
        )

        # Verify the shape was created successfully
        assert shape is not None
        assert shape.obj is not None

    def test_drop_arm(self):
        arm_thick = 5

        app = OpenCascadeApp()
        app.new_document()
        wp = app.work_plane("XY")

        # arm: loft rectangle at hub to rectangle at dropped tip
        arm = (
            wp.move_to(-4.5, 0)
            .line_to(-4.5, 20)
            .line_to(-8, 45)
            .three_point_arc((0, 53), (8, 45))
            .line_to(4.5, 20)
            .line_to(4.5, 0)
            .three_point_arc((0, -4.5), (-4.5, 0))
            .extrude(arm_thick)
        )

        hole = wp.move_to(0, 45).circle(4).extrude(arm_thick)

        arm.cut(hole)
        arm.to_stl("test_drop_arm_occ.stl")

    # PNG Rendering Tests
    # These tests verify the to_png() method which renders 3D shapes to 2D images
    # from different orthographic camera angles using OpenCascade's HLR algorithm.

    def test_to_png_rendering(self):
        """Test PNG rendering from different camera angles."""
        import os

        app = OpenCascadeApp()
        wp = app.work_plane("XY")

        # Create a simple L-shaped object
        shape = (
            wp.move_to(0, 0)
            .line_to(10, 0)
            .line_to(10, 5)
            .line_to(5, 5)
            .line_to(5, 15)
            .line_to(0, 15)
            .line_to(0, 0)
            .extrude(3)
        )

        test_files = [
            "test_render_x.png",
            "test_render_y.png",
            "test_render_z.png",
        ]

        try:
            # Test rendering from different views
            shape.to_png("test_render_x.png", "X", backend="pyvista")  # Right view
            shape.to_png("test_render_y.png", "Y", backend="pyvista")  # Front view
            shape.to_png("test_render_z.png", "Z", backend="pyvista")  # Top view

            # Verify at least one output file was created for each view
            # (either PNG if cairosvg is available, or SVG as fallback)
            assert os.path.exists("test_render_x.png") or os.path.exists(
                "test_render_x.svg"
            ), "X view output file should exist"
            assert os.path.exists("test_render_y.png") or os.path.exists(
                "test_render_y.svg"
            ), "Y view output file should exist"
            assert os.path.exists("test_render_z.png") or os.path.exists(
                "test_render_z.svg"
            ), "Z view output file should exist"

            # Test case-insensitive view parameter
            shape.to_png("test_render_x.png", "x")  # lowercase should work

            # Test invalid view parameter
            with pytest.raises(ValueError, match="View must be"):
                shape.to_png("test_invalid.png", "Invalid")

            with pytest.raises(ValueError, match="View must be"):
                shape.to_png("test_invalid.png", "W")

        finally:
            # Clean up test files
            for file in test_files:
                if os.path.exists(file):
                    os.remove(file)

    def test_to_png_cube(self):
        """Test PNG rendering with a simple cube."""
        import os

        app = OpenCascadeApp()
        wp = app.work_plane("XY")

        # Create a simple 10x10x10 cube
        cube = (
            wp.move_to(0, 0)
            .line_to(10, 0)
            .line_to(10, 10)
            .line_to(0, 10)
            .line_to(0, 0)
            .extrude(10)
        )

        test_file = "test_cube_top_view.png"
        fallback_file = "test_cube_top_view.svg"

        try:
            # Render top view
            cube.to_png(test_file, "Z")

            # Verify output file exists (PNG or SVG fallback)
            assert os.path.exists(test_file) or os.path.exists(
                fallback_file
            ), "Output file should be created"

            # If PNG was created, check it has reasonable size
            if os.path.exists(test_file):
                file_size = os.path.getsize(test_file)
                assert file_size > 100, "PNG file should have reasonable size"

        finally:
            # Clean up
            for file in [test_file, fallback_file]:
                if os.path.exists(file):
                    os.remove(file)

    def test_to_png_with_hole(self):
        """Test PNG rendering with a shape that has a hole (cut operation)."""
        import os

        app = OpenCascadeApp()
        wp = app.work_plane("XY")

        # Create a rectangle with a circular hole
        outer = wp.rect(20, 20).extrude(5)
        hole = wp.move_to(10, 10).circle(3).extrude(5)
        outer.cut(hole)

        test_file = "test_shape_with_hole.png"
        fallback_file = "test_shape_with_hole.svg"

        try:
            # Render from top view to see the hole
            outer.to_png(test_file, "Z")

            # Verify output exists
            assert os.path.exists(test_file) or os.path.exists(
                fallback_file
            ), "Output file should be created for shape with hole"

        finally:
            # Clean up
            for file in [test_file, fallback_file]:
                if os.path.exists(file):
                    os.remove(file)

    def test_shield_cad(self):
        from rapidcadpy import OpenCascadeApp

        # Initialize Inventor application
        app = OpenCascadeApp()
        app.new_document()

        # Sketch 1
        wp1 = app.work_plane("XY")

        wp1.move_to(-0.1692273, -0.06081982).three_point_arc(
            (-0.113369, -0.185601), (0.00293384, -0.25746143)
        ).three_point_arc((0.118158, -0.186268), (0.1692273, -0.06081982)).line_to(
            0.1692273, 0.16610546
        ).three_point_arc(
            (0.079251, 0.182199), (-0.00043091, 0.22698241)
        ).three_point_arc(
            (-0.079712, 0.181706), (-0.1692273, 0.16375488)
        ).line_to(
            -0.1692273, -0.06081982
        )
        # Extrude feature 1
        shape1 = wp1.extrude(0.03175, "NewBodyFeatureOperation")
