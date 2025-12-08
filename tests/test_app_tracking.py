"""Test that the app properly tracks workplanes and shapes."""

import pytest
from rapidcadpy.integrations.ocp.app import OpenCascadeOcpApp


def test_app_tracks_workplanes():
    """Test that the app tracks all created workplanes."""
    app = OpenCascadeOcpApp()

    # Initially should have no workplanes
    assert app.workplane_count() == 0
    assert len(app.get_workplanes()) == 0

    # Create some workplanes
    wp1 = app.work_plane("XY")
    assert app.workplane_count() == 1

    wp2 = app.work_plane("XZ")
    assert app.workplane_count() == 2

    wp3 = app.work_plane("YZ")
    assert app.workplane_count() == 3

    # Verify we can get all workplanes
    workplanes = app.get_workplanes()
    assert len(workplanes) == 3
    assert wp1 in workplanes
    assert wp2 in workplanes
    assert wp3 in workplanes


def test_app_tracks_shapes():
    """Test that the app tracks all created shapes."""
    app = OpenCascadeOcpApp()

    # Initially should have no shapes
    assert app.shape_count() == 0
    assert len(app.get_shapes()) == 0

    # Create a workplane and extrude to create a shape
    wp = app.work_plane("XY")
    shape1 = wp.rect(10, 10).close().extrude(5)

    assert app.shape_count() == 1
    assert shape1 in app.get_shapes()

    # Create another shape
    wp2 = app.work_plane("XY")
    shape2 = wp2.circle(5).close().extrude(3)

    assert app.shape_count() == 2
    shapes = app.get_shapes()
    assert len(shapes) == 2
    assert shape1 in shapes
    assert shape2 in shapes


def test_boolean_operations_dont_create_new_shapes():
    """Test that boolean operations modify in-place and don't register new shapes."""
    app = OpenCascadeOcpApp()

    # Create two shapes
    wp1 = app.work_plane("XY")
    shape1 = wp1.rect(10, 10).close().extrude(5)

    wp2 = app.work_plane("XY")
    shape2 = wp2.move_to(5, 5).rect(5, 5).close().extrude(5)

    # Should have 2 shapes
    assert app.shape_count() == 2

    # Perform a cut operation (in-place modification)
    result = shape1.cut(shape2)

    # Should still have 2 shapes (no new shape created, modified in-place)
    assert app.shape_count() == 2
    assert result is shape1  # Verify it's the same object modified in-place


def test_workplanes_reference_their_app():
    """Test that workplanes maintain a reference to their app."""
    app = OpenCascadeOcpApp()

    wp = app.work_plane("XY")
    assert wp.app is app


def test_shapes_reference_their_app():
    """Test that shapes maintain a reference to their app."""
    app = OpenCascadeOcpApp()

    wp = app.work_plane("XY")
    shape = wp.rect(10, 10).close().extrude(5)

    assert shape.app is app


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
