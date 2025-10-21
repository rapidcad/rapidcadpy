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
    def test_drop_arm_old(self):
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
        wp1.revolve(6.283185307179586, "Z", "JoinBodyFeatureOperation")


    def test_offset_sketch_plane(self):
        from rapidcadpy import InventorApp

        # Initialize Inventor application
        app = InventorApp()
        app.new_document()

        # Sketch 1
        wp1 = app.work_plane("XY")

        wp1.move_to(0.0, 0.0).line_to(0.0, 2.2).line_to(10.7, 2.2).line_to(10.7, 3.0).line_to(13.165, 3.0).line_to(13.165, 3.65).line_to(15.365, 3.65).line_to(15.365, 3.0).line_to(17.83, 3.0).line_to(17.83, 0.0).line_to(0.0, 0.0)

        # Revolve feature 1
        shape1 = wp1.revolve(6.283185307179586, 'X', 'NewBodyFeatureOperation')

        # Sketch 2
        wp2 = app.work_plane("XY", offset=2.2)

        wp2.move_to(0.85, 0.0).line_to(9.85, 0.0)
        wp2.move_to(1.45, -0.6).line_to(9.25, -0.6).three_point_arc((9.85, 0.0), (9.25, 0.6)).line_to(1.45, 0.6).three_point_arc((0.85, 0.0), (1.45, -0.6))

        # Extrude feature 2
        shape2 = wp2.extrude(-0.5, 'Cut', symmetric=False)

    def test_complex_shaft(self):
        from rapidcadpy import InventorApp

        # Initialize Inventor application
        app = InventorApp()
        app.new_document()

        # Sketch 1
        wp1 = app.work_plane("XY")

        wp1.move_to(0.0, 0.0).line_to(0.0, 2.25).line_to(2.165, 2.25).line_to(2.165, 2.9).line_to(3.765, 2.9).line_to(3.765, 2.25).line_to(5.93, 2.25).line_to(5.93, 1.6).line_to(12.63, 1.6).line_to(12.63, 0.0).line_to(0.0, 0.0)

        # Revolve feature 1
        shape1 = wp1.revolve(6.283185307179586, 'X', 'NewBodyFeatureOperation')

        # Sketch 2
        wp2 = app.work_plane("XY")

        wp2.move_to(0.38, 2.125).line_to(0.565, 2.125).line_to(0.565, 2.25).line_to(0.38, 2.25).line_to(0.38, 2.125)

        # Revolve feature 2
        shape2 = wp2.revolve(6.283185307179586, 'X', 'Cut')

        # Sketch 3
        wp3 = app.work_plane("XY")

        wp3.move_to(2.165, 2.45).line_to(2.185, 2.375359)
        wp3.move_to(2.026962, 2.22).line_to(1.915, 2.25).line_to(1.915, 2.45).line_to(2.165, 2.45)
        wp3.move_to(2.026962, 2.22).three_point_arc((2.160525, 2.242195), (2.185, 2.375359))

        # Revolve feature 3
        shape3 = wp3.revolve(6.283185307179586, 'X', 'Cut')

        # Sketch 4
        wp4 = app.work_plane("XY")

        wp4.move_to(5.55, 2.125).line_to(5.365, 2.125).line_to(5.365, 2.25).line_to(5.55, 2.25).line_to(5.55, 2.125)

        # Revolve feature 4
        shape4 = wp4.revolve(6.283185307179586, 'X', 'Cut')

        # Sketch 5
        wp5 = app.work_plane("XY")

        wp5.move_to(3.765, 2.45).line_to(3.745, 2.375359).three_point_arc((3.769475, 2.242195), (3.903038, 2.22)).line_to(4.015, 2.25).line_to(4.015, 2.45).line_to(3.765, 2.45)

        # Revolve feature 5
        shape5 = wp5.revolve(6.283185307179586, 'X', 'Cut')

        # Sketch 6
        wp6 = app.work_plane("XY", offset=1.6)

        wp6.move_to(6.48, 0.0).line_to(12.08, 0.0)
        wp6.move_to(6.98, -0.5).line_to(11.58, -0.5).three_point_arc((12.08, 0.0), (11.58, 0.5)).line_to(6.98, 0.5).three_point_arc((6.48, 0.0), (6.98, -0.5))

        # Extrude feature 6
        shape6 = wp6.extrude(-0.5, 'Cut', symmetric=False)

        # Sketch 7
        wp7 = app.work_plane("XY")

        wp7.move_to(5.93, 1.8).line_to(5.91, 1.725359).three_point_arc((5.934475, 1.592195), (6.068038, 1.57)).line_to(6.18, 1.6).line_to(6.18, 1.8).line_to(5.93, 1.8)

        # Revolve feature 7
        shape7 = wp7.revolve(6.283185307179586, 'X', 'Cut')

    def test_complex_shaft_2(self):
        from rapidcadpy import InventorApp

        # Initialize Inventor application
        app = InventorApp()
        app.new_document()

        # Sketch 1
        wp1 = app.work_plane("XY")

        wp1.move_to(0.0, 0.0).line_to(0.0, 1.6).line_to(8.4, 1.6).line_to(8.4, 2.25).line_to(10.0, 2.25).line_to(10.0, 2.75).line_to(12.465, 2.75).line_to(12.465, 3.25).line_to(14.265, 3.25).line_to(14.265, 2.75).line_to(16.73, 2.75).line_to(16.73, 0.0).line_to(0.0, 0.0)

        # Revolve feature 1
        shape1 = wp1.revolve(6.283185307179586, 'X', 'NewBodyFeatureOperation')

        # Sketch 2
        wp2 = app.work_plane("XY", offset=1.6)

        wp2.move_to(0.7, 0.0).line_to(7.7, 0.0)
        wp2.move_to(1.2, -0.5).line_to(7.2, -0.5).three_point_arc((7.7, 0.0), (7.2, 0.5)).line_to(1.2, 0.5).three_point_arc((0.7, 0.0), (1.2, -0.5))

        # Extrude feature 2
        shape2 = wp2.extrude(-0.5, 'Cut', symmetric=False)

        # Sketch 3
        wp3 = app.work_plane("XY")

        wp3.move_to(8.4, 1.8).line_to(8.42, 1.725359)
        wp3.move_to(8.261962, 1.57).line_to(8.15, 1.6).line_to(8.15, 1.8).line_to(8.4, 1.8)
        wp3.move_to(8.261962, 1.57).three_point_arc((8.395525, 1.592195), (8.42, 1.725359))

        # Revolve feature 3
        shape3 = wp3.revolve(6.283185307179586, 'X', 'Cut')

        # Sketch 4
        wp4 = app.work_plane("XY")

        wp4.move_to(10.0, 2.45).line_to(10.02, 2.375359)
        wp4.move_to(9.861962, 2.22).line_to(9.75, 2.25).line_to(9.75, 2.45).line_to(10.0, 2.45)
        wp4.move_to(9.861962, 2.22).three_point_arc((9.995525, 2.242195), (10.02, 2.375359))

        # Revolve feature 4
        shape4 = wp4.revolve(6.283185307179586, 'X', 'Cut')

        # Sketch 5
        wp5 = app.work_plane("XY")

        wp5.move_to(10.45, 2.6).line_to(10.665, 2.6).line_to(10.665, 2.75).line_to(10.45, 2.75).line_to(10.45, 2.6)

        # Revolve feature 5
        shape5 = wp5.revolve(6.283185307179586, 'X', 'Cut')

        # Sketch 6
        wp6 = app.work_plane("XY")

        wp6.move_to(12.465, 2.95).line_to(12.485, 2.875359)
        wp6.move_to(12.326962, 2.72).line_to(12.215, 2.75).line_to(12.215, 2.95).line_to(12.465, 2.95)
        wp6.move_to(12.326962, 2.72).three_point_arc((12.460525, 2.742195), (12.485, 2.875359))

        # Revolve feature 6
        shape6 = wp6.revolve(6.283185307179586, 'X', 'Cut')

        # Sketch 7
        wp7 = app.work_plane("XY")

        wp7.move_to(16.28, 2.6).line_to(16.065, 2.6).line_to(16.065, 2.75).line_to(16.28, 2.75).line_to(16.28, 2.6)

        # Revolve feature 7
        shape7 = wp7.revolve(6.283185307179586, 'X', 'Cut')

        # Sketch 8
        wp8 = app.work_plane("XY")

        wp8.move_to(14.265, 2.95).line_to(14.245, 2.875359).three_point_arc((14.269475, 2.742195), (14.403038, 2.72)).line_to(14.515, 2.75).line_to(14.515, 2.95).line_to(14.265, 2.95)

        # Revolve feature 8
        shape8 = wp8.revolve(6.283185307179586, 'X', 'Cut')

    def test_drop_arm(self):
        from rapidcadpy import InventorApp

        # Initialize Inventor application
        app = InventorApp()
        app.new_document()

        arm_thick = 6

        wp = app.work_plane("XY")

        # arm: loft rectangle at hub to rectangle at dropped tip
        arm = wp.move_to(-4.5, 0).line_to(-4.5,20).line_to(-8,45).three_point_arc((0,53),(8,45)).line_to(4.5,20).line_to(4.5, 0).three_point_arc((0, -4.5), (-4.5,0)).extrude(arm_thick)

        hole = wp.move_to(0, 45).circle(4).extrude(arm_thick)

        part = arm.cut(hole)

        part.to_ipt("/Users/elias.berger/rapidcad_old/backend/exports/drop_arm.ipt")
