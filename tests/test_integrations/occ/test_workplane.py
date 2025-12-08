import pytest
from rapidcadpy.integrations.ocp.app import OpenCascadeOcpApp
from rapidcadpy.integrations.ocp.workplane import OccWorkplane


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create a temporary directory for test outputs."""
    # Create a subdirectory for workplane tests
    test_dir = tmp_path_factory.mktemp("workplane_tests")
    return test_dir


def test_to_png_simple_sketch(tmp_path):
    """Test rendering a simple 2D sketch to PNG."""
    # Create a simple rectangular sketch
    app = OpenCascadeOcpApp()
    workplane = app.work_plane("XY")
    sketch = (
        workplane.move_to(0, 0).line_to(10, 0).line_to(10, 10).line_to(0, 10).close()
    )

    # Render to PNG before extrusion
    output_file = tmp_path / "test_sketch.png"
    sketch.to_png(str(output_file))

    # Verify file was created
    assert output_file.exists()
    assert output_file.stat().st_size > 0

    # Can still extrude after rendering
    sketch.extrude(2)


def test_to_png_with_circle(tmp_path):
    """Test rendering a sketch with a circle."""
    workplane = OccWorkplane(app=None)
    sketch = workplane.move_to(5, 5).circle(3).close()

    output_file = tmp_path / "test_circle.png"
    sketch.to_png(str(output_file), width=600, height=600)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_to_png_with_arc(tmp_path):
    """Test rendering a sketch with an arc."""
    workplane = OccWorkplane(app=None)
    sketch = workplane.move_to(0, 0).three_point_arc((5, 5), (10, 0)).close()

    output_file = tmp_path / "test_arc.png"
    sketch.to_png(str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_to_png_complex_sketch(tmp_path):
    """Test rendering a complex sketch with multiple primitives."""
    app = OpenCascadeOcpApp()
    workplane = app.work_plane("XY")
    # Create and close a rectangle sketch
    sketch = (
        workplane.move_to(0, 0).line_to(20, 0).line_to(20, 10).line_to(0, 10).close()
    )

    output_file = tmp_path / "test_complex.png"
    sketch.to_png(str(output_file), width=1000, height=800, margin=0.15)

    assert output_file.exists()
    assert output_file.stat().st_size > 0

    # Can still extrude after rendering
    sketch.extrude(2)


def test_to_png_empty_sketch():
    """Test that rendering an empty sketch raises ValueError."""
    workplane = OccWorkplane(app=None)

    with pytest.raises(ValueError, match="No sketch edges available for rendering"):
        # Try to render a sketch that was never created
        try:
            workplane.close()  # This will fail because no pending shapes
        except ValueError:
            pass  # Expected - can't close without shapes

    # Alternative: workplane with no pending shapes
    with pytest.raises(ValueError, match="Cannot create wire: no edges in sketch"):
        workplane.circle(5)  # Add a shape
        workplane._pending_shapes.clear()  # Clear it to simulate empty
        workplane.close()  # This should fail


def test_to_png_custom_dimensions(tmp_path):
    """Test rendering with custom width and height."""
    workplane = OccWorkplane(app=None)
    sketch = workplane.rect(15, 8, centered=False).close()

    output_file = tmp_path / "test_custom.png"
    sketch.to_png(str(output_file), width=1200, height=900, margin=0.2)

    assert output_file.exists()
    # File size should be reasonable for a 1200x900 image
    assert output_file.stat().st_size > 1000


def test_implicit_close_and_extrude():
    # 1. Initialize app and workplane
    app = OpenCascadeOcpApp()
    wp = app.work_plane("XY")

    # 2. Define dimensions
    span = 300.0
    height = 60.0
    truss_height = 20.0
    truss_width = 20.0

    # Sketch and extrude the diagonal truss members
    diag_truss_sketch = wp.move_to(0, truss_height).line_to(span, height - truss_height)
    diag_truss_sketch.extrude(truss_width)


def test_two_workplanes():
    from rapidcadpy.integrations.ocp.app import OpenCascadeOcpApp

    app = OpenCascadeOcpApp()

    # Define variables
    span = 300.0
    width = 80.0
    max_height = 20.0
    top_chord_thickness = 3.0
    bottom_chord_thickness = 3.0
    vertical_thickness = 2.5
    diagonal_thickness = 2.5
    num_bays = 6
    bay_spacing = span / num_bays

    # Create top chord beam
    wp_top = app.work_plane("XY", offset=max_height - top_chord_thickness)
    top_chord = (
        wp_top.move_to(0, 0)
        .line_to(span, 0)
        .line_to(span, width)
        .line_to(0, width)
        .close()
        .extrude(top_chord_thickness)
    )

    # Create bottom chord beam
    wp_bottom = app.work_plane("XY", offset=0)
    bottom_chord = (
        wp_bottom.move_to(0, 0)
        .line_to(span, 0)
        .line_to(span, width)
        .line_to(0, width)
        .close()
        .extrude(bottom_chord_thickness)
    )

    # Create vertical members
    verticals = []
    for i in range(num_bays + 1):
        x_pos = i * bay_spacing
        wp_vert = app.work_plane("XZ", offset=width / 2)
        vert_height = max_height - top_chord_thickness - bottom_chord_thickness
        vertical = (
            wp_vert.move_to(x_pos, bottom_chord_thickness)
            .line_to(x_pos + vertical_thickness, bottom_chord_thickness)
            .line_to(x_pos + vertical_thickness, bottom_chord_thickness + vert_height)
            .line_to(x_pos, bottom_chord_thickness + vert_height)
            .close()
            .extrude(width, mode="center")
        )
        verticals.append(vertical)
    app.show_3d()
