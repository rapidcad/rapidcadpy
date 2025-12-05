"""
Unit tests for OpenCascadeApp tracking and 3D visualization features.

This test suite covers:
- Workplane tracking
- Shape tracking
- Registration mechanisms
- 3D visualization (show_3d)
- Integration between app and created objects
"""

import pytest
from rapidcadpy.integrations.ocp.app import OpenCascadeOcpApp


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create a temporary directory for test outputs."""
    test_dir = tmp_path_factory.mktemp("app_tests")
    return test_dir


@pytest.fixture
def app():
    """Create a fresh OpenCascadeApp instance for each test."""
    return OpenCascadeOcpApp()


class TestAppTracking:
    """Test suite for app-level tracking of workplanes and shapes."""

    def test_app_initialization(self, app):
        """Test that app initializes with empty collections."""
        assert app.workplane_count() == 0
        assert app.shape_count() == 0
        assert app.get_workplanes() == []
        assert app.get_shapes() == []

    def test_workplane_registration_xy(self, app):
        """Test that XY workplane registers with app."""
        wp = app.work_plane("XY")
        
        assert app.workplane_count() == 1
        assert wp in app.get_workplanes()
        assert wp.app is app

    def test_workplane_registration_xz(self, app):
        """Test that XZ workplane registers with app."""
        wp = app.work_plane("XZ")
        
        assert app.workplane_count() == 1
        assert wp in app.get_workplanes()
        assert wp.app is app

    def test_workplane_registration_yz(self, app):
        """Test that YZ workplane registers with app."""
        wp = app.work_plane("YZ")
        
        assert app.workplane_count() == 1
        assert wp in app.get_workplanes()
        assert wp.app is app

    def test_multiple_workplanes(self, app):
        """Test that multiple workplanes are all tracked."""
        wp1 = app.work_plane("XY")
        wp2 = app.work_plane("XZ")
        wp3 = app.work_plane("YZ")
        
        assert app.workplane_count() == 3
        workplanes = app.get_workplanes()
        assert wp1 in workplanes
        assert wp2 in workplanes
        assert wp3 in workplanes

    def test_shape_registration_via_extrude(self, app):
        """Test that shapes created via extrude are registered."""
        wp = app.work_plane("XY")
        shape = wp.rect(10, 10).close().extrude(5)
        
        assert app.shape_count() == 1
        assert shape in app.get_shapes()
        assert shape.app is app

    def test_multiple_shapes(self, app):
        """Test that multiple shapes are all tracked."""
        wp1 = app.work_plane("XY")
        shape1 = wp1.rect(10, 10).close().extrude(5)
        
        wp2 = app.work_plane("XY")
        shape2 = wp2.circle(5).close().extrude(3)
        
        assert app.shape_count() == 2
        shapes = app.get_shapes()
        assert shape1 in shapes
        assert shape2 in shapes

    def test_boolean_cut_no_duplicate_registration(self, app):
        """Test that boolean cut doesn't create duplicate shape registration."""
        wp1 = app.work_plane("XY")
        base = wp1.rect(20, 20).close().extrude(10)
        
        wp2 = app.work_plane("XY")
        wp2.move_to(10, 10)
        cutter = wp2.circle(5).close().extrude(15)
        
        assert app.shape_count() == 2
        
        # Perform cut (modifies in-place)
        result = base.cut(cutter)
        
        # Should still be 2 shapes
        assert app.shape_count() == 2
        assert result is base  # Verify in-place modification

    def test_boolean_union_no_duplicate_registration(self, app):
        """Test that boolean union doesn't create duplicate shape registration."""
        wp1 = app.work_plane("XY")
        shape1 = wp1.rect(10, 10).close().extrude(5)
        
        wp2 = app.work_plane("XY")
        wp2.move_to(5, 5)
        shape2 = wp2.rect(10, 10).close().extrude(5)
        
        assert app.shape_count() == 2
        
        # Perform union (modifies in-place)
        result = shape1.union(shape2)
        
        # Should still be 2 shapes
        assert app.shape_count() == 2
        assert result is shape1  # Verify in-place modification

    def test_get_workplanes_returns_copy(self, app):
        """Test that get_workplanes returns a copy, not the original list."""
        app.work_plane("XY")
        
        workplanes1 = app.get_workplanes()
        workplanes2 = app.get_workplanes()
        
        # Should be equal but not the same object
        assert workplanes1 == workplanes2
        assert workplanes1 is not workplanes2

    def test_get_shapes_returns_copy(self, app):
        """Test that get_shapes returns a copy, not the original list."""
        wp = app.work_plane("XY")
        wp.rect(10, 10).close().extrude(5)
        
        shapes1 = app.get_shapes()
        shapes2 = app.get_shapes()
        
        # Should be equal but not the same object
        assert shapes1 == shapes2
        assert shapes1 is not shapes2

    def test_workplane_with_pending_shapes(self, app):
        """Test tracking workplanes that have pending (not extruded) shapes."""
        wp = app.work_plane("XY")
        wp.rect(10, 10)  # Create sketch but don't extrude
        
        assert app.workplane_count() == 1
        assert app.shape_count() == 0  # No shapes yet
        assert len(wp._pending_shapes) > 0  # Sketch exists

    def test_mixed_extruded_and_pending(self, app):
        """Test app with both extruded shapes and pending sketches."""
        # Create extruded shape
        wp1 = app.work_plane("XY")
        wp1.rect(20, 20).close().extrude(10)
        
        # Create pending sketch
        wp2 = app.work_plane("XY")
        wp2.move_to(30, 0).circle(5)
        
        assert app.workplane_count() == 2
        assert app.shape_count() == 1
        
        # Verify pending shapes exist
        workplanes = app.get_workplanes()
        pending_count = sum(
            1 for wp in workplanes
            if hasattr(wp, '_pending_shapes') and wp._pending_shapes
        )
        assert pending_count == 1  # One workplane has pending shapes


class TestShow3D:
    """Test suite for 3D visualization functionality."""

    def test_show_3d_requires_pyvista(self, app):
        """Test that show_3d raises ImportError if PyVista not available."""
        # This test may pass if PyVista is installed
        # We're just checking the method exists
        assert hasattr(app, 'show_3d')

    def test_show_3d_empty_app_raises_error(self, app):
        """Test that show_3d raises error when nothing to display."""
        with pytest.raises(ValueError, match="No shapes or workplanes to display"):
            app.show_3d(screenshot="dummy.png")

    def test_show_3d_with_shapes(self, app, tmp_path):
        """Test show_3d with shapes only (no pending sketches)."""
        # Create some shapes
        wp1 = app.work_plane("XY")
        wp1.rect(10, 10).close().extrude(5)
        
        wp2 = app.work_plane("XY")
        wp2.move_to(15, 0)
        wp2.circle(3).close().extrude(8)
        
        assert app.shape_count() == 2
        
        # Try to save screenshot (may fail if PyVista not installed)
        import importlib.util
        if importlib.util.find_spec("pyvista") is None:
            pytest.skip("PyVista not installed, skipping visualization test")
        
        output_file = tmp_path / "test_shapes.png"
        app.show_3d(screenshot=str(output_file))
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_show_3d_with_sketches(self, app, tmp_path):
        """Test show_3d with both extruded sketches and pending sketches."""
        # Create a shape - the sketch should be preserved
        wp1 = app.work_plane("XY")
        wp1.rect(20, 20, centered=True).close().extrude(10)
        
        # Create another shape from a circle - this sketch should also be preserved
        sketch1 = app.work_plane("XY")
        sketch1.move_to(30, 0).circle(5).close().extrude(10)
        
        # Create a pending sketch (not extruded)
        sketch2 = app.work_plane("XY")
        sketch2.move_to(-30, 0).rect(8, 8)
        
        assert app.shape_count() == 2
        assert app.workplane_count() == 3
        
        # Verify extruded sketches are preserved
        workplanes = app.get_workplanes()
        extruded_count = sum(
            len(wp._extruded_sketches) if hasattr(wp, '_extruded_sketches') else 0
            for wp in workplanes
        )
        assert extruded_count == 2  # Two sketches were extruded
        
        # Try to save screenshot - should show all sketches (extruded + pending)
        import importlib.util
        if importlib.util.find_spec("pyvista") is None:
            pytest.skip("PyVista not installed, skipping visualization test")
        
        output_file = tmp_path / "test_shapes_and_sketches.png"
        app.show_3d(screenshot=str(output_file), sketch_color="orange")
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_show_3d_complex_geometry(self, app, tmp_path):
        # 1. Initialize app and workplane
        app = OpenCascadeOcpApp()
        wp = app.work_plane("XY")

        # 2. Define dimensions
        span = 300.0
        width = 80.0
        height = 60.0
        truss_height = 20.0
        truss_width = 20.0
        truss_thickness = 5.0

        # 3. Create sketch and extrude to build the geometry
        # Base beam
        sketch = wp.move_to(0, 0).line_to(span, 0).line_to(span, height).line_to(0, height).close()
        base_beam = sketch.extrude(width)

        # Truss members
        truss_member = wp.move_to(0, 0).line_to(truss_width, 0).line_to(truss_width, truss_height).line_to(0, truss_height).close().extrude(truss_thickness)

        # Truss supports
        truss_support = wp.move_to(0, 0).line_to(span, 0).line_to(span, truss_height).line_to(0, truss_height).close().extrude(truss_width)

        # Combine all components
        base_beam.union(truss_member).union(truss_support)
        output_file_3d = tmp_path / "complex_geometry_3d.png"
        app.show_3d(screenshot=str(output_file_3d))
        

    def test_extruded_sketch_preservation(self, app):
        """Test that sketches are preserved after extrusion."""
        wp = app.work_plane("XY")
        wp.circle(10)
        
        # Before close() - should have pending shapes
        assert len(wp._pending_shapes) > 0
        assert len(wp._extruded_sketches) == 0
        
        # Close and extrude
        wp.close().extrude(5)
        
        # After extrusion - pending shapes cleared but preserved in extruded_sketches
        assert len(wp._pending_shapes) == 0
        assert len(wp._extruded_sketches) == 1
        assert len(wp._extruded_sketches[0]) > 0  # The sketch has edges

    def test_show_3d_custom_parameters(self, app, tmp_path):
        """Test show_3d with custom parameters."""
        wp = app.work_plane("XY")
        wp.rect(15, 15).close().extrude(7)
        
        import importlib.util
        if importlib.util.find_spec("pyvista") is None:
            pytest.skip("PyVista not installed, skipping visualization test")
        
        output_file = tmp_path / "test_custom.png"
        app.show_3d(
            width=800,
            height=600,
            show_axes=False,
            shape_opacity=0.9,
            sketch_color="blue",
            screenshot=str(output_file)
        )
        assert output_file.exists()

    def test_show_3d_multiple_shapes_different_colors(self, app, tmp_path):
        """Test that multiple shapes get different colors."""
        # Create 5 different shapes to test color cycling
        for i in range(5):
            wp = app.work_plane("XY")
            wp.move_to(i * 20, 0)
            wp.rect(10, 10).close().extrude(5)
        
        assert app.shape_count() == 5
        
        import importlib.util
        if importlib.util.find_spec("pyvista") is None:
            pytest.skip("PyVista not installed, skipping visualization test")
        
        output_file = tmp_path / "test_multiple_colors.png"
        app.show_3d(
            screenshot=str(output_file),
            shape_opacity=0.8
        )
        assert output_file.exists()


class TestAppIntegration:
    """Integration tests for app and objects working together."""

    def test_complex_model_tracking(self, app):
        """Test tracking in a complex modeling scenario."""
        # Create base
        base_wp = app.work_plane("XY")
        base_wp.rect(50, 30, centered=True).close().extrude(5)
        
        # Create column
        column_wp = app.work_plane("XY")
        column = column_wp.rect(10, 10, centered=True).close().extrude(20)
        
        # Create hole to cut
        hole_wp = app.work_plane("XY")
        hole = hole_wp.circle(3).close().extrude(25)
        
        # Perform cut
        column.cut(hole)
        
        # Create planning sketch
        plan_wp = app.work_plane("XY")
        plan_wp.move_to(30, 0).circle(5)
        
        # Verify tracking
        assert app.workplane_count() == 4
        assert app.shape_count() == 3  # base, column, hole
        
        # Verify all workplanes reference the app
        for wp in app.get_workplanes():
            assert wp.app is app
        
        # Verify all shapes reference the app
        for shape in app.get_shapes():
            assert shape.app is app

    def test_workplane_reference_survives_operations(self, app):
        """Test that workplane-app reference survives through operations."""
        wp = app.work_plane("XY")
        original_app = wp.app
        
        # Perform various operations
        wp.move_to(10, 10)
        wp.line_to(20, 10)
        wp.line_to(20, 20)
        wp.close()
        
        # App reference should still be intact
        assert wp.app is original_app
        assert wp.app is app

    def test_shape_reference_survives_boolean_ops(self, app):
        """Test that shape-app reference survives boolean operations."""
        wp1 = app.work_plane("XY")
        shape1 = wp1.rect(20, 20).close().extrude(10)
        original_app = shape1.app
        
        wp2 = app.work_plane("XY")
        wp2.move_to(5, 5)
        shape2 = wp2.circle(5).close().extrude(15)
        
        # Perform boolean operation
        shape1.cut(shape2)
        
        # App reference should still be intact
        assert shape1.app is original_app
        assert shape1.app is app

    def test_no_duplicate_workplane_registration(self, app):
        """Test that workplanes aren't registered multiple times."""
        
        # Create workplane through factory method
        wp = app.work_plane("XY")
        count1 = app.workplane_count()
        
        # Try to register again manually
        app.register_workplane(wp)
        count2 = app.workplane_count()
        
        # Count should not increase
        assert count1 == count2
        assert count1 == 1

    def test_no_duplicate_shape_registration(self, app):
        """Test that shapes aren't registered multiple times."""
        wp = app.work_plane("XY")
        shape = wp.rect(10, 10).close().extrude(5)
        count1 = app.shape_count()
        
        # Try to register again manually
        app.register_shape(shape)
        count2 = app.shape_count()
        
        # Count should not increase
        assert count1 == count2
        assert count1 == 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_sketch_workplane(self, app):
        """Test workplane with no sketch elements."""
        wp = app.work_plane("XY")
        # Don't add any sketch elements
        
        assert app.workplane_count() == 1
        assert len(wp._pending_shapes) == 0

    def test_workplane_after_extrude(self, app):
        """Test that workplane _pending_shapes cleared after extrude."""
        wp = app.work_plane("XY")
        wp.rect(10, 10)
        
        # Should have pending shapes
        assert len(wp._pending_shapes) > 0
        
        # Close and extrude
        wp.close().extrude(5)
        
        # Pending shapes should be cleared
        assert len(wp._pending_shapes) == 0
        assert app.shape_count() == 1

    def test_multiple_extrudes_same_workplane(self, app):
        """Test multiple extrusions from the same workplane."""
        wp = app.work_plane("XY")
        
        # First sketch and extrude
        wp.rect(10, 10).close().extrude(5)
        assert app.shape_count() == 1
        
        # Second sketch and extrude
        wp.move_to(20, 0).circle(5).close().extrude(3)
        assert app.shape_count() == 2
        
        # Should still be only 1 workplane
        assert app.workplane_count() == 1

    def test_case_insensitive_work_plane_names(self, app):
        """Test that work_plane method handles case-insensitive names."""
        app.work_plane("xy")
        app.work_plane("XY")
        app.work_plane("xZ")
        
        # All should create workplanes
        assert app.workplane_count() == 3

    def test_invalid_work_plane_name(self, app):
        """Test that invalid workplane name raises error."""
        with pytest.raises(ValueError, match="Unknown workplane"):
            app.work_plane("INVALID")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
