"""
Unit tests for the fluent API with Inventor backend.

This module tests the fluent API integration with the Inventor backend,
including creating simple geometries and exporting to .ipt format.
"""

import pytest

from rapidcadpy.integrations.inventor.app import InventorApp


class TestFluentAPIInventorBackend:
    """Test cases for the fluent API with Inventor backend."""

    @pytest.mark.skipif(
        not pytest.importorskip("win32com", reason="win32com not available"),
        reason="win32com not available",
    )
    def test_simple_cube_creation_and_export(self):
        """Test creating a simple cube using fluent API and exporting to .ipt."""
        # Create a simple cube using fluent API
        app = InventorApp()
        app.new_document()
        workplane = app.work_plane("XY")
        cube = workplane.rect(10, 10).extrude(10)
        cube.to_stl("test_simple_cube_inventor.stl")

    @pytest.mark.skipif(
        not pytest.importorskip("win32com", reason="win32com not available"),
        reason="win32com not available",
    )
    def test_drop_arm(self):
        arm_thick = 5

        app = InventorApp()
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

    @pytest.mark.skipif(
        not pytest.importorskip("win32com", reason="win32com not available"),
        reason="win32com not available",
    )
    def test_revolve_simple(self):
        app = InventorApp()
        app.new_document()

        # Sketch 1
        wp1 = app.work_plane("YZ")

        wp1.move_to(0.0, 1.770599).line_to(0.318508, 1.770599).line_to(
            0.318508, 0.532619
        ).line_to(0.527386, 0.532619).line_to(0.527386, 0.259112).line_to(
            0.389785, 0.259112
        ).line_to(0.389785, -0.021593).line_to(0.528091, -0.021593).line_to(
            0.528091, -0.332441
        ).line_to(0.389785, -0.330076).line_to(0.382138, -0.777418).line_to(
            0.0, -0.777418
        ).line_to(0.0, 1.770599)

        # Revolve feature 1
        # wp1.extrude(0.1)  # Thin extrusion to create a profile
        wp1.revolve(6.283185307179586, (0.0, 0.0), "JoinBodyFeatureOperation")


    def test_offset_sketch_plane(self):
        from rapidcadpy import InventorApp

        # Initialize Inventor application
        app = InventorApp(headless=False)
        app.new_document()

        # Sketch 1
        wp1 = app.work_plane("XZ")

        wp1.move_to(0.0, 0.0).line_to(0.977069, 0.0).line_to(0.977069, -1.027451).line_to(0.0, -1.027451).line_to(0.0, 0.0)

        # Extrude feature 1
        shape1 = wp1.extrude(0.254, 'NewBodyFeatureOperation')

        # Sketch 2
        wp2 = app.work_plane(
            origin=(-0.488534, 0.254, -0.513726),
            normal=(0.0, 1.0, 0.0),
            app=app
        )

        wp2.move_to(-0.514625, 0.466042).circle(0.23446)

        # Extrude feature 2
        wp2.extrude(0.254, 'JoinBodyFeatureOperation')