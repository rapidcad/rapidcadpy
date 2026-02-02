"""IPE beam section presets.

This module provides parametric IPE sections as 2D profile factories.

Usage:
    from .components import profiles
    wp = app.work_plane("XY")
    profile = profiles.ipe("IPE80").sketch(wp)
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Section2D
from ...sketch2d import Sketch2D


@dataclass(frozen=True)
class IPESection(Section2D):
    name: str
    depth: float
    flange_width: float
    flange_thickness: float
    web_thickness: float

    def sketch(self, wp, *, x: float = 0.0, y: float = 0.0) -> "Sketch2D":
        """Sketch the IPE profile centered on (x, y) in workplane coordinates.
        Args:
            wp: The workplane to sketch on
            x: X coordinate of section center
            y: Y coordinate of section center
        Returns:
            The resulting Sketch2D with the IPE profile
        """

        half_depth = self.depth / 2.0
        half_flange = self.flange_width / 2.0
        half_web = self.web_thickness / 2.0

        top_outer = y + half_depth
        top_inner = top_outer - self.flange_thickness

        bottom_outer = y - half_depth
        bottom_inner = bottom_outer + self.flange_thickness

        # Clockwise from top-left outer corner
        wp.move_to(x - half_flange, top_outer)
        wp.line_to(x + half_flange, top_outer)
        wp.line_to(x + half_flange, top_inner)
        wp.line_to(x + half_web, top_inner)
        wp.line_to(x + half_web, bottom_inner)
        wp.line_to(x + half_flange, bottom_inner)
        wp.line_to(x + half_flange, bottom_outer)
        wp.line_to(x - half_flange, bottom_outer)
        wp.line_to(x - half_flange, bottom_inner)
        wp.line_to(x - half_web, bottom_inner)
        wp.line_to(x - half_web, top_inner)
        wp.line_to(x - half_flange, top_inner)

        return wp.close()


_IPE_PRESETS: dict[str, IPESection] = {
    "IPE80": IPESection(
        name="IPE80",
        depth=80.0,
        flange_width=46.0,
        flange_thickness=5.2,
        web_thickness=3.8,
    ),
    "IPE100": IPESection(
        name="IPE100",
        depth=100.0,
        flange_width=55.0,
        flange_thickness=5.7,
        web_thickness=4.1,
    ),
    "IPE120": IPESection(
        name="IPE120",
        depth=120.0,
        flange_width=64.0,
        flange_thickness=6.2,
        web_thickness=4.4,
    ),
}


def ipe(name: str) -> IPESection:
    """Get a preset IPE section by name.
    Args:
        name: The name of the IPE profile (e.g., "IPE80")
        Returns:
            The corresponding IPESection instance
    """
    key = name.strip().upper()
    if key not in _IPE_PRESETS:
        available = ", ".join(sorted(_IPE_PRESETS.keys()))
        raise ValueError(f"Unknown IPE profile '{name}'. Available: {available}")
    return _IPE_PRESETS[key]


def list_ipe() -> list[str]:
    return sorted(_IPE_PRESETS.keys())
