"""
Tests for Shape functionality including volume calculation and material properties.
"""

import pytest
from rapidcadpy.integrations.ocp.app import OpenCascadeOcpApp


@pytest.fixture
def app():
    """Create a fresh OpenCascadeApp instance for each test."""
    return OpenCascadeOcpApp()


class TestShapeVolume:
    """Test suite for shape volume calculations."""

    def test_box_volume(self, app):
        """Test volume calculation for a simple box."""
        wp = app.work_plane("XY")
        box = wp.rect(10, 10).close().extrude(5)
        
        expected_volume = 10 * 10 * 5  # 500
        calculated_volume = box.volume()
        
        assert abs(calculated_volume - expected_volume) < 0.01, \
            f"Expected volume {expected_volume}, got {calculated_volume}"

    def test_cylinder_volume(self, app):
        """Test volume calculation for a cylinder."""
        import math
        
        wp = app.work_plane("XY")
        radius = 5
        height = 10
        cylinder = wp.circle(radius).close().extrude(height)
        
        expected_volume = math.pi * radius * radius * height
        calculated_volume = cylinder.volume()
        
        # Allow for small numerical errors
        assert abs(calculated_volume - expected_volume) < 0.1, \
            f"Expected volume {expected_volume:.2f}, got {calculated_volume:.2f}"

    def test_complex_shape_volume(self, app):
        """Test volume calculation after boolean cut operation."""
        # Create base box
        wp_base = app.work_plane("XY")
        base = wp_base.rect(20, 20, centered=True).close().extrude(10)
        base_volume = base.volume()
        
        # Create hole cylinder
        wp_hole = app.work_plane("XY")
        hole = wp_hole.circle(3).close().extrude(15)
        hole_volume = hole.volume()
        
        # Perform cut operation
        result = base.cut(hole)
        result_volume = result.volume()
        
        # Volume after cut should be approximately base - hole
        # (hole extends beyond base, so only intersection is removed)
        expected_min = base_volume - hole_volume
        
        assert result_volume > 0, "Result volume should be positive"
        assert result_volume < base_volume, "Result volume should be less than original"
        assert result_volume > expected_min, "Result volume should account for partial hole"

    def test_union_volume(self, app):
        """Test volume calculation after boolean union operation."""
        # Create two separate boxes
        wp1 = app.work_plane("XY")
        box1 = wp1.rect(10, 10, centered=True).close().extrude(5)
        volume1 = box1.volume()
        
        wp2 = app.work_plane("XY")
        box2 = wp2.move_to(5, 0).rect(10, 10, centered=True).close().extrude(5)
        volume2 = box2.volume()
        
        # Union the boxes
        result = box1.union(box2)
        result_volume = result.volume()
        
        # Union volume should be less than sum (due to overlap)
        assert result_volume > 0, "Union volume should be positive"
        assert result_volume < (volume1 + volume2), "Union should be less than sum due to overlap"
        assert result_volume > max(volume1, volume2), "Union should be larger than either box"

    def test_union_multiple_shapes(self, app):
        """Test union operation with a list of multiple shapes."""
        # Create base box
        wp_base = app.work_plane("XY")
        base = wp_base.rect(10, 10, centered=True).close().extrude(5)
        base_volume = base.volume()
        
        # Create multiple small boxes to union
        shapes_to_union = []
        for i in range(3):
            wp = app.work_plane("XY")
            box = wp.move_to(5 + i * 5, 0).rect(5, 5, centered=True).close().extrude(5)
            shapes_to_union.append(box)
        
        # Union all shapes at once
        result = base.union(shapes_to_union)
        result_volume = result.volume()
        
        # Result should be larger than base
        assert result_volume > base_volume, "Union with multiple shapes should increase volume"
        assert result_volume > 0, "Result volume should be positive"

    def test_union_list_vs_sequential(self, app):
        """Test that union with list gives same result as sequential unions."""
        # Create base and additional shapes
        wp1 = app.work_plane("XY")
        base1 = wp1.rect(10, 10, centered=True).close().extrude(5)
        
        wp2 = app.work_plane("XY")
        base2 = wp2.rect(10, 10, centered=True).close().extrude(5)
        
        wp_a = app.work_plane("XY")
        shape_a = wp_a.move_to(5, 0).rect(5, 5).close().extrude(5)
        
        wp_b = app.work_plane("XY")
        shape_b = wp_b.move_to(0, 5).rect(5, 5).close().extrude(5)
        
        wp_c = app.work_plane("XY")
        shape_c = wp_c.move_to(-5, 0).rect(5, 5).close().extrude(5)
        
        # Method 1: Union with list
        result1 = base1.union([shape_a, shape_b, shape_c])
        volume1 = result1.volume()
        
        # Method 2: Sequential unions (need to recreate shapes)
        wp_a2 = app.work_plane("XY")
        shape_a2 = wp_a2.move_to(5, 0).rect(5, 5).close().extrude(5)
        
        wp_b2 = app.work_plane("XY")
        shape_b2 = wp_b2.move_to(0, 5).rect(5, 5).close().extrude(5)
        
        wp_c2 = app.work_plane("XY")
        shape_c2 = wp_c2.move_to(-5, 0).rect(5, 5).close().extrude(5)
        
        result2 = base2.union(shape_a2).union(shape_b2).union(shape_c2)
        volume2 = result2.volume()
        
        # Volumes should be approximately equal
        assert abs(volume1 - volume2) < 1.0, \
            f"List union ({volume1:.2f}) should match sequential union ({volume2:.2f})"


class TestShapeMaterial:
    """Test suite for shape material properties."""

    def test_material_assignment(self, app):
        """Test that material can be assigned to a shape."""
        wp = app.work_plane("XY")
        shape = wp.rect(10, 10).close().extrude(5)
        
        # Initially should be None
        assert shape.material is None
        
        # Assign material
        shape.material = "Steel"
        assert shape.material == "Steel"

    def test_material_reassignment(self, app):
        """Test that material can be changed."""
        wp = app.work_plane("XY")
        shape = wp.rect(10, 10).close().extrude(5)
        
        shape.material = "Aluminum"
        assert shape.material == "Aluminum"
        
        shape.material = "Copper"
        assert shape.material == "Copper"

    def test_material_preserved_after_cut(self, app):
        """Test that material is preserved after boolean cut operation."""
        wp1 = app.work_plane("XY")
        base = wp1.rect(20, 20, centered=True).close().extrude(10)
        base.material = "Aluminum"
        
        wp2 = app.work_plane("XY")
        hole = wp2.circle(3).close().extrude(15)
        
        # Perform cut (modifies base in-place)
        result = base.cut(hole)
        
        # Material should be preserved
        assert result.material == "Aluminum"

    def test_material_preserved_after_union(self, app):
        """Test that material is preserved after boolean union operation."""
        wp1 = app.work_plane("XY")
        shape1 = wp1.rect(10, 10).close().extrude(5)
        shape1.material = "Steel"
        
        wp2 = app.work_plane("XY")
        shape2 = wp2.move_to(5, 0).rect(10, 10).close().extrude(5)
        shape2.material = "Copper"
        
        # Union (modifies shape1 in-place)
        result = shape1.union(shape2)
        
        # Material of first shape should be preserved
        assert result.material == "Steel"

    def test_multiple_shapes_different_materials(self, app):
        """Test that multiple shapes can have different materials."""
        wp1 = app.work_plane("XY")
        box = wp1.rect(10, 10).close().extrude(5)
        box.material = "Steel"
        
        wp2 = app.work_plane("XY")
        cylinder = wp2.circle(3).close().extrude(8)
        cylinder.material = "Aluminum"
        
        wp3 = app.work_plane("XY")
        another_box = wp3.rect(5, 5).close().extrude(2)
        another_box.material = "Copper"
        
        assert box.material == "Steel"
        assert cylinder.material == "Aluminum"
        assert another_box.material == "Copper"


class TestShapeVolumeAndMaterial:
    """Test suite for combined volume and material functionality."""

    def test_volume_with_material(self, app):
        """Test that volume calculation works with material assigned."""
        wp = app.work_plane("XY")
        shape = wp.rect(15, 8).close().extrude(3)
        shape.material = "Copper"
        
        expected_volume = 15 * 8 * 3  # 360
        calculated_volume = shape.volume()
        
        assert shape.material == "Copper"
        assert abs(calculated_volume - expected_volume) < 0.01

    def test_material_does_not_affect_volume(self, app):
        """Test that changing material doesn't affect volume calculation."""
        wp = app.work_plane("XY")
        shape = wp.rect(10, 10).close().extrude(5)
        
        volume_before = shape.volume()
        
        shape.material = "Steel"
        volume_with_steel = shape.volume()
        
        shape.material = "Aluminum"
        volume_with_aluminum = shape.volume()
        
        assert abs(volume_before - volume_with_steel) < 0.01
        assert abs(volume_before - volume_with_aluminum) < 0.01


class TestPipe:
    def test_simple_pipe(self):
        """
        Test the pipe() functionality
        """

        from rapidcadpy.integrations.ocp.app import OpenCascadeOcpApp


        app = OpenCascadeOcpApp()
        wp = app.work_plane("XY")

        # Create a path using line_to
        sketch = wp.move_to(0,0).line_to(100, 0).line_to(100, 100).line_to(0, 100).close()


        # Create a pipe with 10mm diameter

        pipe_shape = sketch.pipe(diameter=10.0)
        print("✓ Pipe created successfully!")
        print(f"  Pipe volume: {pipe_shape.volume():.2f}")

    def test_truss_pipe(self):
        from rapidcadpy.integrations.ocp.app import OpenCascadeOcpApp
        app = OpenCascadeOcpApp()
        height = 1
        width = 10
        wp = app.work_plane("XZ", offset=0)


        # create bridge body
        wp.move_to(0,0).line_to(20,0).line_to(20,-height).line_to(0,-height).close().extrude(width)
        
        # create left truss
        wp.move_to(0,0).line_to(5,5).line_to(10, 0).line_to(15,5).line_to(20,0).close().pipe(diameter=1)

        # create right truss
        wp_right = app.work_plane("XZ", offset=width)
        wp_right.move_to(0,0).line_to(5,5).line_to(10, 0).line_to(15,5).line_to(20,0).close().pipe(diameter=1)
        app.show_3d()
