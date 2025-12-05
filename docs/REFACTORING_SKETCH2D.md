# Workplane Refactoring: Sketch2D Architecture

## Summary

This document describes a major architectural refactoring of RapidCAD-Py that separates sketch construction from extrusion operations. The refactoring introduces a new `Sketch2D` class that represents a finalized 2D sketch ready for 3D operations.

## Motivation

Previously, the workplane was responsible for both sketch construction AND extrusion:

```python
# OLD API - confusing responsibility
wp = app.work_plane("XY")
wp.rect(10, 10)  # Builds sketch edges
wp.close()       # Closes the sketch but doesn't construct face
wp.extrude(5)    # NOW constructs face and extrudes
```

This had several problems:
1. **Unclear semantics**: `close()` didn't actually finalize anything
2. **Mixed responsibilities**: Workplane handled both 2D sketching and 3D operations
3. **Hard to extend**: Adding new 3D operations (revolve, loft, etc.) would clutter workplane

## New Architecture

### Sketch2D Class

We introduced a new `Sketch2D` abstract class that represents a constructed 2D face:

**File**: `rapidcadpy/sketch2d.py`
```python
class Sketch2D(ABC):
    """A finalized 2D sketch ready for 3D operations."""
    
    def __init__(self, face: Any, workplane: Any, app: Optional["App"] = None):
        self._face = face
        self._workplane = workplane
        self.app = app
    
    @abstractmethod
    def extrude(self, distance: float) -> Shape: ...
    
    @abstractmethod
    def to_png(self, file_name: str, ...) -> None: ...
```

**Implementation**: `rapidcadpy/integrations/occ/sketch2d.py`
```python
class OccSketch2D(Sketch2D):
    """OpenCASCADE implementation of Sketch2D."""
    
    def extrude(self, distance: float) -> OccShape:
        # Constructs prism from face along workplane normal
        ...
```

### Updated Workplane Behavior

**Key Change**: `close()` now constructs the face and returns a `Sketch2D` object:

```python
# rapidcadpy/integrations/occ/workplane.py
def close(self) -> "OccSketch2D":
    """
    Close the current sketch and construct the face.
    Returns a Sketch2D object that can be extruded.
    """
    # Close the loop if needed
    if self._loop_start is not None:
        # Auto-close the sketch
        ...
    
    # Construct face from pending edges
    face = self._make_face()
    
    # Preserve sketch edges for visualization
    self._clear_pending_shapes()
    
    # Return Sketch2D object
    return OccSketch2D(face=face, workplane=self, app=self.app)
```

**Key Point**: `extrude()` method was REMOVED from `Workplane` class.

## New API Pattern

### Before (Old API)
```python
wp = app.work_plane("XY")
wp.rect(10, 10)
wp.close()
shape = wp.extrude(5)  # Workplane does extrusion
```

### After (New API)
```python
wp = app.work_plane("XY")
sketch = wp.rect(10, 10).close()  # close() returns Sketch2D
shape = sketch.extrude(5)          # Sketch2D does extrusion
```

### Fluent Chaining
The new API supports fluent chaining:

```python
# All in one chain
shape = wp.rect(10, 10).close().extrude(5)

# Or save intermediate sketch
sketch = wp.circle(5).close()
sketch.to_png("my_sketch.png")  # Render the 2D sketch
shape = sketch.extrude(10)       # Then extrude it
```

## Benefits

### 1. Clear Separation of Concerns
- **Workplane**: 2D sketch construction (lines, arcs, circles)
- **Sketch2D**: 3D operations (extrude, future: revolve, loft, sweep)

### 2. Better Semantics
- `close()` now actually finalizes something (constructs the face)
- Face construction happens at a clear, explicit point
- Extrusion is clearly a 3D operation on a 2D sketch

### 3. Extensibility
Future 3D operations can be added to `Sketch2D` without cluttering `Workplane`:

```python
# Future possibilities
sketch = wp.circle(5).close()
shape1 = sketch.extrude(10)
shape2 = sketch.revolve(360)  # Revolve around axis
shape3 = sketch.sweep(path)   # Sweep along path
```

### 4. Better Error Messages
Face construction now happens in `close()`, so errors are reported earlier:

```python
sketch = wp.rect(10, 10).close()  # Face construction error here
# vs old way where error happened in extrude()
```

### 5. Sketch Preservation
Sketches can be saved, visualized, and reused:

```python
sketch = wp.circle(5).close()
sketch.to_png("design.png")        # Visualize the 2D sketch
shape1 = sketch.extrude(5)         # Extrude once
shape2 = sketch.extrude(10)        # Extrude again with different height
```

## Migration Guide

### Simple Cases
**Before:**
```python
wp.rect(10, 10).extrude(5)
```

**After:**
```python
wp.rect(10, 10).close().extrude(5)
```

Just add `.close()` before `.extrude()`.

### Multi-step Cases
**Before:**
```python
wp = app.work_plane("XY")
wp.rect(10, 10)
wp.close()
shape = wp.extrude(5)
```

**After:**
```python
wp = app.work_plane("XY")
sketch = wp.rect(10, 10).close()  # close() returns Sketch2D
shape = sketch.extrude(5)
```

### Complex Sketches
**Before:**
```python
wp = app.work_plane("XY")
wp.move_to(0, 0)
wp.line_to(10, 0)
wp.line_to(10, 10)
wp.line_to(0, 10)
wp.close()  # Just closes the loop
shape = wp.extrude(5)
```

**After:**
```python
wp = app.work_plane("XY")
sketch = (wp.move_to(0, 0)
          .line_to(10, 0)
          .line_to(10, 10)
          .line_to(0, 10)
          .close())  # Constructs face AND returns Sketch2D
shape = sketch.extrude(5)
```

## Implementation Details

### Files Changed

1. **`rapidcadpy/sketch2d.py`** (NEW)
   - Abstract `Sketch2D` class

2. **`rapidcadpy/integrations/occ/sketch2d.py`** (NEW)
   - `OccSketch2D` implementation with `extrude()` and `to_png()`

3. **`rapidcadpy/workplane.py`** (MODIFIED)
   - Added `_loop_start` attribute initialization
   - Updated `close()` abstract method signature to return `Sketch2D`
   - Removed `extrude()` abstract method

4. **`rapidcadpy/integrations/occ/workplane.py`** (MODIFIED)
   - Updated `close()` to construct face and return `OccSketch2D`
   - Removed `extrude()` method (moved to `OccSketch2D`)

5. **All test files** (MODIFIED)
   - Updated to use `.close().extrude()` pattern

### Error Handling Improvements

Face construction errors now happen in `close()` with detailed messages:

```python
try:
    sketch = wp.rect(10, 10).close()
except ValueError as e:
    # "Face construction failed with 4 edge(s): ..."
    # Includes troubleshooting steps
```

## Testing

All 31 tests in `test_app_tracking.py` pass with the new API:

```bash
$ pytest tests/test_integrations/occ/test_app_tracking.py -v
===============================================================================
31 passed in 8.86s
===============================================================================
```

## Future Enhancements

With this architecture, we can easily add:

1. **Revolve operation**
   ```python
   sketch = wp.circle(5).close()
   shape = sketch.revolve(axis=(0,0,1), angle=360)
   ```

2. **Loft operation**
   ```python
   sketch1 = wp1.circle(10).close()
   sketch2 = wp2.circle(5).close()
   shape = Sketch2D.loft([sketch1, sketch2])
   ```

3. **Sweep operation**
   ```python
   profile = wp.rect(2, 2).close()
   path = create_path_curve(...)
   shape = profile.sweep(path)
   ```

4. **Sketch operations**
   ```python
   sketch1 = wp.circle(10).close()
   sketch2 = wp.circle(5).close()
   combined = sketch1.subtract(sketch2)  # 2D boolean
   shape = combined.extrude(5)
   ```

## Backward Compatibility

This is a **breaking change**. Code using the old API must be updated to add `.close()` before `.extrude()`.

An automated migration script could be:

```python
import re
content = re.sub(r'\.rect\(([^)]*)\)\.extrude\(', r'.rect(\1).close().extrude(', content)
content = re.sub(r'\.circle\(([^)]*)\)\.extrude\(', r'.circle(\1).close().extrude(', content)
```

## Conclusion

This refactoring provides:
- ✅ Clearer separation of 2D and 3D operations
- ✅ Better semantics (`close()` actually finalizes the sketch)
- ✅ Extensibility for future 3D operations
- ✅ Improved error messages
- ✅ Sketch reusability and visualization

The new `Sketch2D` class represents a natural intermediate state between 2D sketching and 3D modeling, making the API more intuitive and powerful.
