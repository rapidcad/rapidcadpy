"""
Unit tests for the fluent API workplane functionality.

This module tests the CadQuery-like fluent API including Workplane, SketchBuilder,
and integration with the existing CAD model structure.
"""

import pytest

from rapidcadpy import Circle, Line, Plane, Vector, Vertex, Workplane
from rapidcadpy.cad import Cad
from rapidcadpy.sketch_extrude import SketchExtrude
from rapidcadpy.workplane import SketchBuilder


class TestWorkplane:
    """Test cases for the Workplane class."""

    @pytest.fixture
    def xy_plane(self):
        """Create a standard XY plane at origin."""
        return Plane(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))

    @pytest.fixture
    def offset_plane(self):
        """Create an offset plane for testing."""
        return Plane(Vector(-0.2, 0.06, 0), Vector(1, 0, 0), Vector(0, 0, 1))

    @pytest.fixture
    def workplane(self, xy_plane):
        """Create a basic workplane."""
        return Workplane(xy_plane)

    def test_workplane_initialization(self, xy_plane):
        """Test that workplane initializes correctly."""
        wp = Workplane(xy_plane)
        assert wp.plane == xy_plane
        assert wp._current_position == Vertex(0, 0)
        assert wp._pending_shapes == []
        assert wp._current_sketch is None

    def test_move_to(self, workplane):
        """Test moveTo functionality."""
        result = workplane.moveTo(1.5, 2.5)

        # Should return self for chaining
        assert result is workplane
        # Should update current position
        assert workplane._current_position.x == 1.5
        assert workplane._current_position.y == 2.5

    def test_circle_creation(self, workplane):
        """Test circle creation."""
        workplane.moveTo(1, 1)
        result = workplane.circle(0.5)

        # Should return self for chaining
        assert result is workplane
        # Should add a circle to pending shapes
        assert len(workplane._pending_shapes) == 1
        assert isinstance(workplane._pending_shapes[0], Circle)

        circle = workplane._pending_shapes[0]
        assert circle.center.x == 1
        assert circle.center.y == 1
        assert circle.radius == 0.5

    def test_line_creation(self, workplane):
        """Test line creation."""
        workplane.moveTo(0, 0)
        result = workplane.line(3, 4)

        # Should return self for chaining
        assert result is workplane
        # Should add a line to pending shapes
        assert len(workplane._pending_shapes) == 1
        assert isinstance(workplane._pending_shapes[0], Line)

        line = workplane._pending_shapes[0]
        assert line.start_point.x == 0
        assert line.start_point.y == 0
        assert line.end_point.x == 3
        assert line.end_point.y == 4

        # Should update current position
        assert workplane._current_position.x == 3
        assert workplane._current_position.y == 4

    def test_line_to_alias(self, workplane):
        """Test that lineTo is an alias for line."""
        workplane.moveTo(1, 1)
        result = workplane.lineTo(2, 2)

        assert result is workplane
        assert len(workplane._pending_shapes) == 1
        assert isinstance(workplane._pending_shapes[0], Line)

    def test_rectangle_creation_centered(self, workplane):
        """Test centered rectangle creation."""
        workplane.moveTo(5, 5)
        result = workplane.rect(4, 2, centered=True)

        # Should return self for chaining
        assert result is workplane
        # Should create 4 lines for rectangle
        assert len(workplane._pending_shapes) == 4

        # All shapes should be lines
        for shape in workplane._pending_shapes:
            assert isinstance(shape, Line)

        # Check rectangle bounds (centered at 5,5 with width=4, height=2)
        # Expected corners: (3,4), (7,4), (7,6), (3,6)
        lines = workplane._pending_shapes

        # Collect all points
        points = []
        for line in lines:
            points.extend(
                [
                    (line.start_point.x, line.start_point.y),
                    (line.end_point.x, line.end_point.y),
                ]
            )

        # Should have points at expected corners
        expected_points = {(3, 4), (7, 4), (7, 6), (3, 6)}
        actual_points = set(points)

        # Check that we have the expected corner points
        assert expected_points.issubset(actual_points)

    def test_rectangle_creation_not_centered(self, workplane):
        """Test non-centered rectangle creation."""
        workplane.moveTo(0, 0)
        result = workplane.rect(2, 3, centered=False)

        assert result is workplane
        assert len(workplane._pending_shapes) == 4

        # Check that rectangle starts at current position
        lines = workplane._pending_shapes
        points = []
        for line in lines:
            points.extend(
                [
                    (line.start_point.x, line.start_point.y),
                    (line.end_point.x, line.end_point.y),
                ]
            )

        # Should have origin point and corner at (2,3)
        actual_points = set(points)
        assert (0, 0) in actual_points
        assert (2, 3) in actual_points

    def test_chaining_operations(self, workplane):
        """Test that operations can be chained together."""
        result = workplane.moveTo(1, 1).circle(0.5).moveTo(2, 2).line(3, 3)

        assert result is workplane
        assert len(workplane._pending_shapes) == 2
        assert isinstance(workplane._pending_shapes[0], Circle)
        assert isinstance(workplane._pending_shapes[1], Line)

    def test_add_workplanes(self, xy_plane):
        """Test adding two workplanes together."""
        wp1 = Workplane(xy_plane).moveTo(0, 0).circle(1)
        wp2 = Workplane(xy_plane).moveTo(2, 2).rect(1, 1)

        result = wp1.add(wp2)

        assert isinstance(result, SketchBuilder)
        assert result.plane == xy_plane
        # Should combine shapes from both workplanes
        assert len(result.shapes) == 5  # 1 circle + 4 lines from rectangle

    def test_direct_extrude(self, workplane):
        """Test direct extrusion from workplane."""
        workplane.moveTo(0, 0).circle(1)
        result = workplane.extrude(2.0)

        assert isinstance(result, Cad)
        assert len(result.construction_history) == 1

        feature = result.construction_history[0]
        assert isinstance(feature, SketchExtrude)
        assert feature.extrude.extent_one == 2.0
        assert len(feature.sketch) == 1

    def test_requested_syntax_example(self):
        """Test the exact syntax requested by the user."""
        wp_sketch0 = Workplane(
            Plane(Vector(-0.2, 0.06, 0), Vector(1, 0, 0), Vector(0, 0, 1))
        )
        loop0 = wp_sketch0.moveTo(0.01, 0).circle(0.01)
        solid0 = wp_sketch0.add(loop0).extrude(0.75)
        solid = solid0

        # Verify the result
        assert isinstance(solid, Cad)
        assert len(solid.construction_history) == 1

        feature = solid.construction_history[0]
        assert isinstance(feature, SketchExtrude)
        assert feature.extrude.extent_one == 0.75

        # Check that the circle was created correctly
        sketch = feature.sketch[0]
        assert len(sketch.outer_wire.edges) == 1
        circle = sketch.outer_wire.edges[0]
        assert isinstance(circle, Circle)
        assert circle.radius == 0.01


class TestSketchBuilder:
    """Test cases for the SketchBuilder class."""

    @pytest.fixture
    def xy_plane(self):
        """Create a standard XY plane at origin."""
        return Plane(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))

    @pytest.fixture
    def circle_shapes(self):
        """Create a list with a circle shape."""
        return [Circle(center=Vertex(0, 0), radius=1)]

    @pytest.fixture
    def line_shapes(self):
        """Create a list with line shapes forming a square."""
        return [
            Line(start_point=Vertex(0, 0), end_point=Vertex(1, 0)),
            Line(start_point=Vertex(1, 0), end_point=Vertex(1, 1)),
            Line(start_point=Vertex(1, 1), end_point=Vertex(0, 1)),
            Line(start_point=Vertex(0, 1), end_point=Vertex(0, 0)),
        ]

    def test_sketch_builder_initialization(self, xy_plane, circle_shapes):
        """Test SketchBuilder initialization."""
        builder = SketchBuilder(xy_plane, circle_shapes)

        assert builder.plane == xy_plane
        assert builder.shapes == circle_shapes

    def test_extrude_circle(self, xy_plane, circle_shapes):
        """Test extruding a circle."""
        builder = SketchBuilder(xy_plane, circle_shapes)
        result = builder.extrude(3.0, "NewBodyFeatureOperation")

        assert isinstance(result, Cad)
        assert len(result.construction_history) == 1

        feature = result.construction_history[0]
        assert isinstance(feature, SketchExtrude)
        assert feature.extrude.extent_one == 3.0
        assert feature.extrude.operation == "NewBodyFeatureOperation"
        assert feature.sketch_plane == xy_plane

        # Check sketch has the circle
        sketch = feature.sketch[0]
        assert hasattr(sketch, "sketch_plane")
        assert sketch.sketch_plane == xy_plane
        assert len(sketch.outer_wire.edges) == 1
        assert isinstance(sketch.outer_wire.edges[0], Circle)

    def test_extrude_lines(self, xy_plane, line_shapes):
        """Test extruding a shape made of lines."""
        builder = SketchBuilder(xy_plane, line_shapes)
        result = builder.extrude(1.5, "CutFeatureOperation")

        assert isinstance(result, Cad)
        feature = result.construction_history[0]
        assert feature.extrude.extent_one == 1.5
        assert feature.extrude.operation == "CutFeatureOperation"

        # Check sketch has all the lines
        sketch = feature.sketch[0]
        assert len(sketch.outer_wire.edges) == 4
        for edge in sketch.outer_wire.edges:
            assert isinstance(edge, Line)

    def test_default_extrude_operation(self, xy_plane, circle_shapes):
        """Test that default extrude operation is applied correctly."""
        builder = SketchBuilder(xy_plane, circle_shapes)
        result = builder.extrude(2.0)  # No operation specified

        feature = result.construction_history[0]
        assert feature.extrude.operation == "NewBodyFeatureOperation"


class TestFluentAPIIntegration:
    """Integration tests for the fluent API with existing CAD functionality."""

    def test_cad_object_has_to_inventor_method(self):
        """Test that CAD objects have the to_inventor method."""
        wp = Workplane(Plane(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0)))
        solid = wp.circle(1).extrude(1)

        # Check that the method exists
        assert hasattr(solid, "to_inventor")
        assert callable(solid.to_inventor)

    def test_step_export_works(self, tmp_path):
        """Test that STEP export works with fluent API generated models."""
        wp = Workplane(Plane(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0)))
        solid = wp.circle(0.5).extrude(1)

        # Create a temporary file path
        step_file = tmp_path / "test_fluent.step"

        # Should not raise an exception
        try:
            solid.to_step(str(step_file))
            # Check that file was created
            assert step_file.exists()
        except Exception as e:
            pytest.fail(f"STEP export failed: {e}")

    def test_complex_workflow(self):
        """Test a complex workflow combining multiple fluent operations."""
        # Create a workplane
        wp = Workplane(Plane(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0)))

        # Create a complex shape with multiple operations
        shape1 = wp.moveTo(-2, -2).rect(4, 4, centered=False)
        shape2 = Workplane(wp.plane).moveTo(0, 0).circle(1)

        # Combine and extrude
        combined = shape1.add(shape2)
        solid = combined.extrude(2.0)

        # Verify the result
        assert isinstance(solid, Cad)
        assert len(solid.construction_history) == 1

        feature = solid.construction_history[0]
        sketch = feature.sketch[0]

        # Should have 5 shapes: 4 lines from rectangle + 1 circle
        assert len(sketch.outer_wire.edges) == 5

        # Count shapes by type
        lines = [e for e in sketch.outer_wire.edges if isinstance(e, Line)]
        circles = [e for e in sketch.outer_wire.edges if isinstance(e, Circle)]

        assert len(lines) == 4
        assert len(circles) == 1

    def test_multiple_extrusions(self):
        """Test creating multiple separate extrusions."""
        plane = Plane(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))

        # Create first solid
        wp1 = Workplane(plane)
        solid1 = wp1.moveTo(0, 0).circle(1).extrude(1)

        # Create second solid
        wp2 = Workplane(plane)
        solid2 = wp2.moveTo(5, 5).rect(2, 2).extrude(0.5)

        # Both should be valid CAD objects
        assert isinstance(solid1, Cad)
        assert isinstance(solid2, Cad)

        # Should have independent construction histories
        assert len(solid1.construction_history) == 1
        assert len(solid2.construction_history) == 1

        # Different extrusion parameters
        assert solid1.construction_history[0].extrude.extent_one == 1
        assert solid2.construction_history[0].extrude.extent_one == 0.5


class TestErrorHandling:
    """Test error handling in the fluent API."""

    def test_invalid_plane_type(self):
        """Test that invalid plane types are handled appropriately."""
        # This should work fine - just testing that it doesn't crash
        wp = Workplane("not_a_plane")
        assert wp.plane == "not_a_plane"

    def test_empty_workplane_extrude(self):
        """Test extruding an empty workplane."""
        wp = Workplane(Plane(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0)))

        # Should create a CAD object even with no shapes
        result = wp.extrude(1.0)
        assert isinstance(result, Cad)

        # Should have empty wire
        feature = result.construction_history[0]
        sketch = feature.sketch[0]
        assert len(sketch.outer_wire.edges) == 0

    def test_zero_radius_circle(self):
        """Test creating a circle with zero radius."""
        wp = Workplane(Plane(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0)))

        # Should not crash
        wp.circle(0)
        assert len(wp._pending_shapes) == 1

        circle = wp._pending_shapes[0]
        assert circle.radius == 0

    def test_negative_extrude_distance(self):
        """Test extruding with negative distance."""
        wp = Workplane(Plane(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0)))
        solid = wp.circle(1).extrude(-1.0)

        assert isinstance(solid, Cad)
        feature = solid.construction_history[0]
        assert feature.extrude.extent_one == -1.0
