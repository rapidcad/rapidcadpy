# 3D Visualization Guide

RapidCAD-Py includes powerful 3D visualization capabilities that allow you to view all shapes and sketches created within an app instance.

## Overview

The `show_3d()` method in `OpenCascadeApp` provides an interactive 3D viewer that displays:
- All extruded shapes with different colors and transparency
- Pending sketches (not yet extruded) as red wireframes in 3D space
- Previously extruded sketches (preserved for visualization) also as wireframes
- Coordinate axes for reference
- Interactive controls for rotating, zooming, and panning

## Requirements

The 3D visualization feature requires PyVista:

```bash
pip install pyvista
```

## Basic Usage

```python
from rapidcadpy.integrations.occ.app import OpenCascadeApp

# Create app
app = OpenCascadeApp()

# Create some shapes
wp1 = app.work_plane("XY")
box = wp1.rect(20, 20, centered=True).extrude(10)

wp2 = app.work_plane("XY")
cylinder = wp2.circle(5).extrude(15)

# Create a sketch (not extruded)
sketch_wp = app.work_plane("XY")
sketch_wp.move_to(25, 0).rect(10, 10)

# Display everything in 3D
app.show_3d()
```

## Features

### Automatic Tracking

The app automatically tracks all workplanes and shapes:

```python
# Check what's been created
print(f"Workplanes: {app.workplane_count()}")
print(f"Shapes: {app.shape_count()}")

# Access the collections
workplanes = app.get_workplanes()
shapes = app.get_shapes()
```

### Customization Options

```python
app.show_3d(
    width=1200,              # Window width in pixels
    height=800,              # Window height in pixels
    show_axes=True,          # Show/hide coordinate axes
    shape_opacity=0.8,       # Shape transparency (0-1)
    sketch_color="red",      # Color for sketches
    show_edges=False,        # Show/hide mesh tessellation lines
    screenshot="output.png"  # Save screenshot instead of showing
)
```

### Interactive Controls

When the 3D viewer opens:
- **Left mouse**: Rotate the view
- **Right mouse**: Pan the view
- **Scroll wheel**: Zoom in/out
- **'r'**: Reset view
- **'s'**: Take screenshot
- **'q'**: Quit viewer

## Examples

### Example 1: Multiple Shapes with Boolean Operations

```python
app = OpenCascadeApp()

# Create base
base_wp = app.work_plane("XY")
base = base_wp.rect(50, 30).extrude(5)

# Create feature to cut
cut_wp = app.work_plane("XY")
cut_wp.move_to(25, 15)
cutter = cut_wp.circle(8).extrude(10)

# Perform cut
base.cut(cutter)

# Add another shape
wp = app.work_plane("XY")
wp.move_to(60, 0)
extra = wp.rect(15, 15).extrude(8)

# Visualize
app.show_3d(shape_opacity=0.9)
```

### Example 2: Sketches and Shapes Together

```python
app = OpenCascadeApp()

# Create extruded shape
wp1 = app.work_plane("XY")
shape = wp1.rect(20, 20, centered=True).extrude(10)

# Create planning sketches (not extruded yet)
sketch1 = app.work_plane("XY")
sketch1.move_to(30, 0).circle(5)

sketch2 = app.work_plane("XY")
sketch2.move_to(-30, 0).rect(10, 10)

# Show everything - sketches appear as red wireframes
app.show_3d(sketch_color="orange")
```

### Example 3: Save Screenshot

```python
app = OpenCascadeApp()

# Build your model
wp = app.work_plane("XY")
shape = wp.rect(30, 30, centered=True).extrude(20)

# Save visualization without showing window
app.show_3d(
    width=1920,
    height=1080,
    show_axes=True,
    screenshot="my_model.png"
)
```

## Color Scheme

Shapes are automatically assigned colors from a palette:
1. Light Blue
2. Light Green
3. Light Yellow
4. Light Coral
5. Light Pink
6. Light Gray
7. Lavender
8. Peach Puff

If you have more than 8 shapes, colors will cycle through the palette.

## Tips

1. **Performance**: For complex models with many shapes, consider using lower opacity values to see overlapping geometry better.

2. **Screenshots**: Use the `screenshot` parameter for automated rendering in scripts or CI/CD pipelines.

3. **Debugging**: The 3D view is excellent for debugging geometry issues - you can see exactly where shapes are positioned and how sketches relate to extruded geometry.

4. **Workflow**: The visualization helps you understand your construction history and plan next steps in model creation.

5. **Clean Rendering**: Set `show_edges=False` (default) for smooth, professional-looking renders without mesh tessellation lines. Set `show_edges=True` if you want to see the underlying mesh structure.

### Smooth vs. Mesh Rendering

```python
# Smooth rendering (default) - clean professional look
app.show_3d(show_edges=False)

# Show mesh edges - useful for understanding tessellation
app.show_3d(show_edges=True)
```

## API Reference

### `show_3d()`

Visualize all shapes and sketches in 3D space.

**Parameters:**
- `width` (int): Window width in pixels (default: 1200)
- `height` (int): Window height in pixels (default: 800)
- `show_axes` (bool): Whether to show coordinate axes (default: True)
- `shape_opacity` (float): Opacity of 3D shapes, 0-1 (default: 0.8)
- `sketch_color` (str): Color for 2D sketches (default: "red")
- `screenshot` (Optional[str]): Path to save screenshot instead of showing interactively
- `show_edges` (bool): Whether to show mesh edges/tessellation lines (default: False)

**Raises:**
- `ImportError`: If PyVista is not installed
- `ValueError`: If no shapes or workplanes to display

**Returns:** None

## See Also

- [Basic Modeling Tutorial](tutorials/basic_modeling.md)
- [Shape Operations](api/cad.md)
- [Workplane Reference](api/workplane.md)
