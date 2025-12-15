"""Reusable parametric 2D section profiles.

A "section" is a *parametric profile factory* that can sketch itself onto a
workplane. Users can then extrude or sweep that profile.

Design goal:
- backend-agnostic: section building is done using the Workplane/Sketch API
  (move_to/line_to/close, arcs later).
- composable: return the produced sketch so it can be used immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Section2D(Protocol):
    """A parametric 2D profile that can be sketched on a workplane."""

    def sketch(self, wp, *, x: float = 0.0, y: float = 0.0):
        """Draw this section on the given workplane and return the resulting sketch."""


@dataclass(frozen=True)
class SectionSpec:
    """Optional base for simple section specs."""

    name: str
