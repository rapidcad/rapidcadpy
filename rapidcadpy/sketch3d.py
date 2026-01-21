"""3D sketch abstractions.

`Sketch3D` represents a *spatial* curve/path (wire) used as a sweep/pipe spine.

This is intentionally minimal:
- it stores 3D curve primitives (like a polyline)
- it can produce a backend-specific wire/path object

Backends (OCC/OCP/etc) provide concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

from rapidcadpy.shape import Shape

Point3D = Tuple[float, float, float]


@dataclass(frozen=True)
class Polyline3D:
    """A simple polyline primitive for 3D sketches."""

    points: Tuple[Point3D, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("Polyline3D requires at least 2 points")


class Sketch3D(ABC):
    """Abstract base class for a 3D path sketch.

    Contract:
    - Instances collect primitives that define a single continuous path.
    - `wire()` returns a backend-native wire/path object.
    """

    def __init__(self, app):
        self.app = app
        self._primitives: List[object] = []
        self._cursor: Point3D | None = None

    def move_to(self, x: float, y: float, z: float) -> "Sketch3D":
        """Set the current cursor/start point for subsequent operations."""
        self._cursor = (float(x), float(y), float(z))
        return self

    def line_to(self, x: float, y: float, z: float) -> "Sketch3D":
        """Draw a straight line from the current cursor to the specified point.

        Args:
            x: X coordinate of the end point
            y: Y coordinate of the end point
            z: Z coordinate of the end point

        Returns:
            Self for method chaining

        Raises:
            ValueError: If no cursor position is set (call move_to first)
        """
        if self._cursor is None:
            raise ValueError("No cursor position set. Call move_to() first.")

        end_point: Point3D = (float(x), float(y), float(z))
        self._primitives.append(Polyline3D((self._cursor, end_point)))
        self._cursor = end_point
        return self

    def polyline(self, points: Sequence[Point3D]) -> "Sketch3D":
        """Append a polyline segment to the sketch.

        If a cursor is set via `move_to`, the cursor point will be prepended
        unless the first polyline point already matches the cursor.
        """
        pts: list[Point3D] = [(float(x), float(y), float(z)) for (x, y, z) in points]
        if self._cursor is not None:
            if len(pts) == 0:
                pts = [self._cursor]
            elif pts[0] != self._cursor:
                pts = [self._cursor, *pts]
        self._primitives.append(Polyline3D(tuple(pts)))
        self._cursor = pts[-1]
        return self

    def close(self) -> "Sketch3D":
        """Close the path by drawing a line back to the first point.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If the sketch has no primitives or cursor position
        """
        if not self._primitives:
            raise ValueError("Cannot close: no primitives in sketch")
        
        # Get the first point from the first primitive
        first_primitive = self._primitives[0]
        if isinstance(first_primitive, Polyline3D):
            first_point = first_primitive.points[0]
        else:
            raise ValueError("Cannot determine first point of sketch")
        
        if self._cursor is None:
            raise ValueError("No cursor position set")
        
        # Only add closing line if we're not already at the start
        if self._cursor != first_point:
            self._primitives.append(Polyline3D((self._cursor, first_point)))
            self._cursor = first_point
        
        return self

    def finish(self) -> "Sketch3D":
        """Finish the sketch and return self for final operations.

        This is a convenience method for fluent API chaining.

        Returns:
            Self
        """
        return self

    @property
    def primitives(self) -> Iterable[object]:
        return tuple(self._primitives)

    @abstractmethod
    def wire(self):
        """Return a backend-native wire/path object."""

    @abstractmethod
    def pipe(
        self,
        diameter: float,
        is_frenet: bool = True,
        transition_mode: str = "right",
    ) -> "Shape":
        """
        Create a pipe (cylindrical sweep) along this 3D sketch (wire) spine.
        args:
            diameter: Diameter of the pipe
            is_frenet: Whether to use Frenet frames for orientation
            transition_mode: Transition mode for profile orientation
        Returns:
            Shape: The resulting pipe shape
        """
        ...

    @abstractmethod
    def sweep(
        self,
        profile: "Any",
        make_solid: bool = True,
        is_frenet: bool = True,
        transition_mode: str = "right",
        auto_align_profile: bool = False,
    ) -> "Shape":
        """Sweep a closed 2D profile sketch along this 3D spine.
        args:
            profile: The 2D profile sketch to sweep along the path
            make_solid: Whether to create a solid from the sweep
            is_frenet: Whether to use Frenet frames for orientation
            transition_mode: Transition mode for profile orientation
            auto_align_profile: Whether to auto-align the profile at the start of the sweep
        Returns:
            Shape: The resulting swept shape"""
        ...