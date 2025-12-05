"""
2D Geometric primitives for sketching.

These classes represent 2D geometric entities in the workplane's coordinate system.
They are converted to 3D geometry when constructing faces.
"""

from typing import Tuple


class Line:
    """A 2D line segment in workplane coordinates."""
    def __init__(self, start: Tuple[float, float], end: Tuple[float, float]):
        self.start = start  # (x, y) in 2D workplane coordinates
        self.end = end      # (x, y) in 2D workplane coordinates


class Circle:
    """A 2D circle in workplane coordinates."""
    def __init__(self, center: Tuple[float, float], radius: float):
        self.center = center  # (x, y) in 2D workplane coordinates
        self.radius = radius


class Arc:
    """A 2D arc defined by three points in workplane coordinates."""
    def __init__(self, start: Tuple[float, float], mid: Tuple[float, float], end: Tuple[float, float]):
        self.start = start  # (x, y) in 2D workplane coordinates
        self.mid = mid      # (x, y) in 2D workplane coordinates
        self.end = end      # (x, y) in 2D workplane coordinates
    