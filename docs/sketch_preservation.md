# Sketch Preservation Feature

## Overview

The 3D visualization now preserves and displays sketches even after they've been extruded, allowing you to see the complete construction history of your model.

## How It Works

When you create a sketch and extrude it:

```python
app = OpenCascadeApp()

wp = app.work_plane("XY")
shape = wp.circle(10).extrude(5)  # The circle sketch is preserved!
```

**Previously:** The circle sketch would be cleared after extrusion and lost.

**Now:** The circle sketch is preserved in `wp._extruded_sketches` and will be visible when you call `app.show_3d()`.

## Benefits

### 1. **Construction History Visibility**
See exactly how your model was built, with all original sketches visible alongside the 3D shapes.

### 2. **Better Debugging**
Identify issues by seeing both the 2D sketch and the resulting 3D extrusion.

### 3. **Design Documentation**
Screenshots automatically show your design intent with both sketches and shapes.

### 4. **Learning Tool**
Great for tutorials and teaching - students can see the relationship between 2D sketches and 3D results.

## Example

```python
from rapidcadpy.integrations.occ.app import OpenCascadeApp

app = OpenCascadeApp()

# Create multiple shapes - all sketches will be preserved
base_wp = app.work_plane("XY")
base = base_wp.rect(30, 30, centered=True).extrude(5)

column_wp = app.work_plane("XY")
column = column_wp.circle(6).extrude(20)

hole_wp = app.work_plane("XY")
hole_wp.move_to(10, 10)
hole = hole_wp.circle(2).extrude(25)

column.cut(hole)

# Visualize - you'll see:
# - 3 colored shapes (base, column, hole)
# - 3 red sketches (the original rectangle and 2 circles)
app.show_3d(
    shape_opacity=0.7,  # Make shapes semi-transparent
    sketch_color="red"
)
```

## What Gets Preserved

✅ **Lines** - All line segments  
✅ **Circles** - Complete circles and arcs  
✅ **Rectangles** - Rectangle sketches  
✅ **Arcs** - Three-point arcs  
✅ **Complex Sketches** - Any combination of the above

## Technical Details

### Data Structure

Each `Workplane` now has two collections:

1. **`_pending_shapes`** - Active sketch being built (cleared after extrusion)
2. **`_extruded_sketches`** - List of lists containing preserved sketch edges

### Storage

```python
# After extrusion:
wp._extruded_sketches = [
    [edge1, edge2, edge3, ...],  # First extruded sketch
    [edge4, edge5, ...],         # Second extruded sketch (if any)
]
```

### Visualization

When `show_3d()` is called:
1. Iterates through all workplanes
2. Renders `_pending_shapes` (not yet extruded)
3. Renders all sketches in `_extruded_sketches` (previously extruded)
4. All sketches shown in the same color (default: red)

## Customization

### Change Sketch Color

```python
app.show_3d(sketch_color="orange")  # Use orange for all sketches
```

### Adjust Shape Opacity

```python
# Make shapes more transparent to see sketches better
app.show_3d(shape_opacity=0.6)
```

### Hide Axes

```python
app.show_3d(show_axes=False)
```

## Performance Considerations

- Minimal memory overhead (stores edge references, not full geometry)
- No impact on modeling performance
- Visualization time increases slightly with many sketches
- For models with 100+ sketches, consider clearing `_extruded_sketches` manually if not needed

## Clearing Preserved Sketches

If you want to clear preserved sketches (e.g., for memory in very large models):

```python
# Clear all extruded sketches from all workplanes
for wp in app.get_workplanes():
    if hasattr(wp, '_extruded_sketches'):
        wp._extruded_sketches.clear()
```

## Use Cases

### 1. Design Review
Show clients both the design intent (sketches) and the result (3D shapes).

### 2. Manufacturing Documentation
Document the construction sequence for fabrication planning.

### 3. Education
Teach CAD concepts by showing the relationship between 2D and 3D.

### 4. Debugging
Verify that sketches are positioned correctly before complex operations.

### 5. Version Control
Include sketch visibility in screenshots for better change tracking.

## Comparison: Before vs After

### Before (Old Behavior)
```python
wp = app.work_plane("XY")
wp.rect(20, 20)
shape = wp.extrude(10)
# wp._pending_shapes is now empty []
# Original sketch is lost
```

### After (New Behavior)
```python
wp = app.work_plane("XY")
wp.rect(20, 20)
shape = wp.extrude(10)
# wp._pending_shapes is empty []
# wp._extruded_sketches contains the rectangle edges
# Sketch visible in show_3d()
```

## Future Enhancements

Potential additions:
- Different colors for different sketches
- Toggle sketch visibility on/off
- Sketch labels with extrusion order
- Time-based animation showing construction sequence
- Export sketches separately to DXF format

## See Also

- [Visualization Guide](visualization_guide.md)
- [Examples: extruded_sketches_demo.py](../examples/extruded_sketches_demo.py)
- [API Reference](api/cad.md)
