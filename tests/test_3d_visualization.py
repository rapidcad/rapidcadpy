"""
Test file for the show_3d() visualization feature.

This demonstrates that the app correctly tracks and visualizes
all shapes and sketches.
"""

import pytest
from rapidcadpy.integrations.ocp.app import OpenCascadeOcpApp


def test_show_3d_requires_content():
    """Test that show_3d raises error when nothing to display."""
    app = OpenCascadeOcpApp()
    
    with pytest.raises(ValueError, match="No shapes or workplanes to display"):
        app.show_3d(screenshot="test.png")


def test_show_3d_with_shapes_only():
    """Test visualization with only shapes (no pending sketches)."""
    app = OpenCascadeOcpApp()
    
    # Create some shapes
    wp1 = app.work_plane("XY")
    wp1.rect(10, 10).close().extrude(5)
    
    wp2 = app.work_plane("XY")
    wp2.move_to(15, 0)
    wp2.circle(3).close().extrude(8)
    
    assert app.shape_count() == 2
    assert app.workplane_count() == 2
    
    # Should work without error (can't test interactive display in CI)
    # Just verify the method exists and can be called
    try:
        import pyvista
        # If pyvista is available, save a screenshot
        app.show_3d(screenshot="test_shapes.png")
        import os
        assert os.path.exists("test_shapes.png")
        os.remove("test_shapes.png")
    except ImportError:
        # If pyvista not available, just verify method exists
        assert hasattr(app, 'show_3d')


def test_show_3d_with_sketches():
    """Test visualization with shapes and pending sketches."""
    app = OpenCascadeOcpApp()
    
    # Create a shape
    wp1 = app.work_plane("XY")
    wp1.rect(20, 20, centered=True).close().extrude(10)
    
    # Create pending sketches (not extruded)
    sketch1 = app.work_plane("XY")
    sketch1.move_to(30, 0).circle(5)
    
    sketch2 = app.work_plane("XY")
    sketch2.move_to(-30, 0).rect(8, 8)
    
    assert app.shape_count() == 1
    assert app.workplane_count() == 3
    
    # Verify sketches have pending shapes
    workplanes = app.get_workplanes()
    pending_count = sum(
        1 for wp in workplanes
        if hasattr(wp, '_pending_shapes') and wp._pending_shapes
    )
    assert pending_count == 2  # Two sketches have pending shapes


def test_app_tracking_integration():
    """Test that app properly tracks all created workplanes and shapes."""
    app = OpenCascadeOcpApp()
    
    # Initially empty
    assert app.workplane_count() == 0
    assert app.shape_count() == 0
    
    # Create first workplane and shape
    wp1 = app.work_plane("XY")
    assert app.workplane_count() == 1
    
    shape1 = wp1.rect(10, 10).close().extrude(5)
    assert app.shape_count() == 1
    
    # Create second workplane and shape
    wp2 = app.work_plane("XZ")
    assert app.workplane_count() == 2
    
    shape2 = wp2.rect(8, 8).close().extrude(3)
    assert app.shape_count() == 2
    
    # Verify we can retrieve them
    shapes = app.get_shapes()
    assert len(shapes) == 2
    assert shape1 in shapes
    assert shape2 in shapes
    
    workplanes = app.get_workplanes()
    assert len(workplanes) == 2
    assert wp1 in workplanes
    assert wp2 in workplanes


def test_boolean_operations_tracking():
    """Test that boolean operations don't create duplicate shape registrations."""
    app = OpenCascadeOcpApp()
    
    # Create two shapes
    wp1 = app.work_plane("XY")
    base = wp1.rect(20, 20).close().extrude(10)
    
    wp2 = app.work_plane("XY")
    wp2.move_to(10, 10)
    cutter = wp2.circle(5).close().extrude(15)
    
    assert app.shape_count() == 2
    
    # Perform boolean operation (modifies in-place)
    result = base.cut(cutter)
    
    # Should still be 2 shapes (cut modifies in-place)
    assert app.shape_count() == 2
    assert result is base


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
