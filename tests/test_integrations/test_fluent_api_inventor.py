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
        wp1.revolve(6.283185307179586, "Z", "JoinBodyFeatureOperation")

    def test_offset_sketch_plane(self):
        from rapidcadpy import InventorApp

        # Initialize Inventor application
        app = InventorApp()
        app.new_document()

        # Sketch 1
        wp1 = app.work_plane("XY")

        wp1.move_to(0.0, 0.0).line_to(0.0, 2.2).line_to(10.7, 2.2).line_to(
            10.7, 3.0
        ).line_to(13.165, 3.0).line_to(13.165, 3.65).line_to(15.365, 3.65).line_to(
            15.365, 3.0
        ).line_to(17.83, 3.0).line_to(17.83, 0.0).line_to(0.0, 0.0)

        # Revolve feature 1
        shape1 = wp1.revolve(6.283185307179586, "X", "NewBodyFeatureOperation")

        # Sketch 2
        wp2 = app.work_plane("XY", offset=2.2)

        wp2.move_to(0.85, 0.0).line_to(9.85, 0.0)
        wp2.move_to(1.45, -0.6).line_to(9.25, -0.6).three_point_arc(
            (9.85, 0.0), (9.25, 0.6)
        ).line_to(1.45, 0.6).three_point_arc((0.85, 0.0), (1.45, -0.6))

        # Extrude feature 2
        shape2 = wp2.extrude(-0.5, "Cut", symmetric=False)

    def test_shield_cad(self):
        from rapidcadpy import InventorApp

        # Initialize Inventor application
        app = InventorApp()
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

    def test_flash_cad(self):
        from rapidcadpy import InventorApp

        # Initialize Inventor application
        app = InventorApp()
        app.new_document()

        # Sketch 1
        wp1 = app.work_plane("XY")

        wp1.move_to(-0.06272749, 0.07966536).line_to(-0.03073996, 0.05362793).line_to(
            -0.02266331, 0.04705364
        ).line_to(0.03514306, 0.0).line_to(0.00856945, 0.0).line_to(
            0.04752207, -0.03956649
        ).line_to(0.02309416, -0.03956649).line_to(0.03900109, -0.05955874).line_to(
            0.04503432, -0.06714148
        ).line_to(0.06788577, -0.09586179).line_to(0.03801842, -0.07127586).line_to(
            0.02978309, -0.06449677
        ).line_to(-0.02933965, -0.01582865).line_to(-0.00266195, -0.01582865).line_to(
            -0.04099881, 0.02310104
        ).line_to(-0.01728528, 0.02310104).line_to(-0.03188096, 0.04126905).line_to(
            -0.03792803, 0.04879616
        ).line_to(-0.06272749, 0.07966536).close()
        wp1.move_to(-0.03792803, 0.04879616).three_point_arc(
            (-0.054894, -0.045989), (0.03801842, -0.07127586)
        ).line_to(0.02978309, -0.06449677).three_point_arc(
            (-0.047894, -0.038926), (-0.03188096, 0.04126905)
        ).line_to(-0.03792803, 0.04879616).close()
        wp1.move_to(-0.03073996, 0.05362793).line_to(
            -0.02266331, 0.04705364
        ).three_point_arc((0.058757, 0.023008), (0.03900109, -0.05955874)).line_to(
            0.04503432, -0.06714148
        ).three_point_arc((0.065789, 0.030037), (-0.03073996, 0.05362793)).close()
        # Extrude feature 1
        shape1 = wp1.extrude(0.00508, "NewBodyFeatureOperation")

    def test_internal_holes(self):
        from rapidcadpy import InventorApp

        # Initialize Inventor application
        app = InventorApp()
        app.new_document()

        # Sketch 1
        wp1 = app.work_plane("XZ")

        wp1.move_to(-0.015, 0.0).circle(0.0075).close()
        wp1.move_to(0.015, 0.0).circle(0.0075).close()
        wp1.move_to(-0.015, -0.01).line_to(0.015, -0.01).three_point_arc((0.015, 0.01), (0.015, 0.01)).line_to(-0.015, 0.01).three_point_arc((-0.015, 0.01), (-0.015, -0.01)).close()
        wp1.move_to(0.0, 0.0).circle(0.0035).close()
        # Extrude feature 1
        shape1 = wp1.extrude(0.003, 'NewBodyFeatureOperation')