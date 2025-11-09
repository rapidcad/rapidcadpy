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
        ).three_point_arc((-0.079712, 0.181706), (-0.1692273, 0.16375488)).line_to(
            -0.1692273, -0.06081982
        )
        # Extrude feature 1
        shape1 = wp1.extrude(0.03175, "NewBodyFeatureOperation")
