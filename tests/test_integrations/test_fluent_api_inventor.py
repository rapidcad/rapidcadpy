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
        wp1.move_to(-0.015, -0.01).line_to(0.015, -0.01).three_point_arc(
            (0.015, 0.01), (0.015, 0.01)
        ).line_to(-0.015, 0.01).three_point_arc((-0.015, 0.01), (-0.015, -0.01)).close()
        wp1.move_to(0.0, 0.0).circle(0.0035).close()
        # Extrude feature 1
        shape1 = wp1.extrude(0.003, "NewBodyFeatureOperation")

    def test_gencad(self):
        from rapidcadpy import InventorApp

        app = InventorApp()

        app.new_document()

        # Generating a workplane for sketch 0
        wp_sketch0 = app.work_plane(origin=(-0.75, 0.0, 0.0), normal=(0.0, 0.0, 1.0))

        loop0 = wp_sketch0.move_to(0.2131578947368421, 0.0).circle(0.2087171052631579)

        solid0 = wp_sketch0.add(loop0).extrude(0.109375)
        solid = solid0

        # Generating a workplane for sketch 1
        wp_sketch1 = app.work_plane(
            origin=(-0.109375, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
        )

        loop1 = wp_sketch1.move_to(0.10657894736842105, 0.0).circle(0.10657894736842105)

        solid1 = wp_sketch1.add(loop1).extrude(0.0625)
        solid = solid.union(solid1)

        # Generating a workplane for sketch 2
        wp_sketch2 = app.work_plane(
            origin=(0.2578125, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
        )

        loop2 = wp_sketch2.move_to(0.09473684210526316, 0.0).circle(0.09276315789473684)

        solid2 = wp_sketch2.add(loop2).extrude(0.0234375)
        solid = solid.union(solid2)

    def test_chamfer_edge(self):
        from rapidcadpy import InventorApp

        # Initialize Inventor application
        app = InventorApp()
        app.new_document()

        # Sketch 1
        wp1 = app.work_plane("XY")

        wp1.move_to(0.0, 0.0).line_to(0.0, 1.7).line_to(9.6, 1.7).line_to(
            9.6, 2.5
        ).line_to(13.465, 2.5).line_to(13.465, 3.2).line_to(15.465, 3.2).line_to(
            15.465, 2.5
        ).line_to(17.73, 2.5).line_to(17.73, 2.1).line_to(19.33, 2.1).line_to(
            19.33, 1.3
        ).line_to(26.33, 1.3).line_to(26.33, 0.0).line_to(0.0, 0.0)

        # Revolve feature 1
        shape1 = wp1.revolve(6.283185307179586, "X", "NewBodyFeatureOperation")

        # Sketch 2
        wp2 = app.work_plane("XY", offset=1.7000000000000002)

        wp2.move_to(0.8, 0.0).line_to(8.8, 0.0)
        wp2.move_to(1.3, -0.5).line_to(8.3, -0.5).three_point_arc(
            (8.8, 0.0), (8.3, 0.5)
        ).line_to(1.3, 0.5).three_point_arc((0.8, 0.0), (1.3, -0.5))

        # Extrude feature 2
        shape2 = wp2.extrude(-0.5, "Cut", symmetric=False)

        # Sketch 3
        wp3 = app.work_plane("XY")

        wp3.move_to(9.6, 1.9).line_to(9.619516, 1.761134)
        wp3.move_to(9.540295, 1.67).line_to(9.472494, 1.67)
        wp3.move_to(9.451788, 1.672726).line_to(9.35, 1.7).line_to(9.35, 1.9).line_to(
            9.6, 1.9
        )
        wp3.move_to(9.540295, 1.67).three_point_arc(
            (9.600672, 1.697515), (9.619516, 1.761134)
        )
        wp3.move_to(9.451788, 1.672726).three_point_arc(
            (9.462052, 1.670684), (9.472494, 1.67)
        )

        # Revolve feature 3
        shape3 = wp3.revolve(6.283185307179586, "X", "Cut")

        # Sketch 4
        wp4 = app.work_plane("XY")

        wp4.move_to(13.465, 2.7).line_to(13.484516, 2.561134)
        wp4.move_to(13.405295, 2.47).line_to(13.337494, 2.47)
        wp4.move_to(13.316788, 2.472726).line_to(13.215, 2.5).line_to(
            13.215, 2.7
        ).line_to(13.465, 2.7)
        wp4.move_to(13.405295, 2.47).three_point_arc(
            (13.465672, 2.497515), (13.484516, 2.561134)
        )
        wp4.move_to(13.316788, 2.472726).three_point_arc(
            (13.327052, 2.470684), (13.337494, 2.47)
        )

        # Revolve feature 4
        shape4 = wp4.revolve(6.283185307179586, "X", "Cut")

        # Sketch 5
        wp5 = app.work_plane("XY", offset=2.5)

        wp5.move_to(9.3, 0.0).line_to(10.9, 0.0)
        wp5.move_to(9.6, -0.3).line_to(10.6, -0.3).three_point_arc(
            (10.9, 0.0), (10.6, 0.3)
        ).line_to(9.6, 0.3).three_point_arc((9.3, 0.0), (9.6, -0.3))

        # Extrude feature 5
        shape5 = wp5.extrude(-0.25, "Cut", symmetric=False)

        # Sketch 6
        wp6 = app.work_plane("XY")

        wp6.move_to(17.28, 2.35).line_to(17.065, 2.35).line_to(17.065, 2.5).line_to(
            17.28, 2.5
        ).line_to(17.28, 2.35)

        # Revolve feature 6
        shape6 = wp6.revolve(6.283185307179586, "X", "Cut")

        # Sketch 7
        wp7 = app.work_plane("XY")

        wp7.move_to(15.465, 2.7).line_to(15.445484, 2.561134).three_point_arc(
            (15.464328, 2.497515), (15.524705, 2.47)
        ).line_to(15.592506, 2.47).three_point_arc(
            (15.602948, 2.470684), (15.613212, 2.472726)
        ).line_to(15.715, 2.5).line_to(15.715, 2.7).line_to(15.465, 2.7)

        # Revolve feature 7
        shape7 = wp7.revolve(6.283185307179586, "X", "Cut")

        # Sketch 8
        wp8 = app.work_plane("XY")

        wp8.move_to(17.73, 2.3).line_to(17.710484, 2.161134).three_point_arc(
            (17.729328, 2.097515), (17.789705, 2.07)
        ).line_to(17.857506, 2.07).three_point_arc(
            (17.867948, 2.070684), (17.878212, 2.072726)
        ).line_to(17.98, 2.1).line_to(17.98, 2.3).line_to(17.73, 2.3)

        # Revolve feature 8
        shape8 = wp8.revolve(6.283185307179586, "X", "Cut")

        # Sketch 9
        wp9 = app.work_plane("XY", offset=1.3)

        wp9.move_to(19.68, 0.0).line_to(25.98, 0.0)
        wp9.move_to(20.08, -0.4).line_to(25.58, -0.4).three_point_arc(
            (25.98, 0.0), (25.58, 0.4)
        ).line_to(20.08, 0.4).three_point_arc((19.68, 0.0), (20.08, -0.4))

        # Extrude feature 9
        shape9 = wp9.extrude(-0.4, "Cut", symmetric=False)

        # Sketch 10
        wp10 = app.work_plane("XY")

        wp10.move_to(19.33, 1.5).line_to(19.310484, 1.361134).three_point_arc(
            (19.329328, 1.297515), (19.389705, 1.27)
        ).line_to(19.457506, 1.27).three_point_arc(
            (19.467948, 1.270684), (19.478212, 1.272726)
        ).line_to(19.58, 1.3).line_to(19.58, 1.5).line_to(19.33, 1.5)

        # Revolve feature 10
        shape10 = wp10.revolve(6.283185307179586, "X", "Cut")

        # Sketch 11
        wp11 = app.work_plane("XY")

        wp11.move_to(19.33, 0.0).line_to(19.33, 0.4).line_to(19.252058, 0.265).line_to(
            19.009571, 0.125
        ).line_to(18.602169, 0.125).line_to(18.53, 0.0).line_to(19.33, 0.0)

        # Revolve feature 11
        shape11 = wp11.revolve(6.283185307179586, "X", "Cut")

        # Chamfered Edges
        app.chamfer_edge(x=19.33, radius=2.1, angle=1.83, distance=0.18)
