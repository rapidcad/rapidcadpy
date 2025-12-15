"""IPN (tapered flange) beam section presets.

IPN sections differ from IPE primarily by having tapered flanges.

Note:
- The current implementation approximates a tapered flange profile using straight
  segments (no arcs/fillets yet). This keeps the section backend-agnostic.
- Dimensions here should be treated as a starting point; if you have a preferred
  table (EN 10365 / DIN 1025-1), we can swap in exact values.

Usage:
    from rapidcadpy.components import profiles
    profile = profiles.ipn("IPN80").sketch(wp)
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidcadpy.components.sections.base import Section2D


@dataclass(frozen=True)
class IPNSection(Section2D):
    name: str
    depth: float
    flange_width: float
    web_thickness: float
    flange_thickness_mid: float
    flange_thickness_edge: float

    def sketch(self, wp, *, x: float = 0.0, y: float = 0.0):
        """Sketch an approximate IPN profile centered on (x, y).

        Approximation: each flange is a trapezoid (taper) with thickness varying from
        `flange_thickness_mid` at the web to `flange_thickness_edge` at the outer edge.
        """

        h2 = self.depth / 2.0
        b2 = self.flange_width / 2.0
        tw2 = self.web_thickness / 2.0

        # Outer y extents
        y_top = y + h2
        y_bot = y - h2

        # Thickness near web vs at flange tip
        t_mid = self.flange_thickness_mid
        t_edge = self.flange_thickness_edge

        # Inner faces of flanges (where web meets)
        y_top_inner = y_top - t_mid
        y_bot_inner = y_bot + t_mid

        # Tapered inner corner at outer edge
        y_top_edge_inner = y_top - t_edge
        y_bot_edge_inner = y_bot + t_edge

        # Clockwise outline starting top-left outer
        wp.move_to(x - b2, y_top)
        wp.line_to(x + b2, y_top)
        wp.line_to(x + b2, y_top_edge_inner)
        wp.line_to(x + tw2, y_top_inner)
        wp.line_to(x + tw2, y_bot_inner)
        wp.line_to(x + b2, y_bot_edge_inner)
        wp.line_to(x + b2, y_bot)
        wp.line_to(x - b2, y_bot)
        wp.line_to(x - b2, y_bot_edge_inner)
        wp.line_to(x - tw2, y_bot_inner)
        wp.line_to(x - tw2, y_top_inner)
        wp.line_to(x - b2, y_top_edge_inner)

        return wp.close()


# --- Presets --------------------------------------------------------------
# These values are approximate placeholders.
# If you provide the exact table you want to use, we'll replace them.
_IPN_PRESETS: dict[str, IPNSection] = {
    "IPN80": IPNSection(
        name="IPN80",
        depth=80.0,
        flange_width=42.0,
        web_thickness=3.9,
        flange_thickness_mid=5.9,
        flange_thickness_edge=4.3,
    ),
    "IPN100": IPNSection(
        name="IPN100",
        depth=100.0,
        flange_width=50.0,
        web_thickness=4.5,
        flange_thickness_mid=6.8,
        flange_thickness_edge=5.0,
    ),
    "IPN120": IPNSection(
        name="IPN120",
        depth=120.0,
        flange_width=58.0,
        web_thickness=5.1,
        flange_thickness_mid=7.7,
        flange_thickness_edge=5.7,
    ),
}


def ipn(name: str) -> IPNSection:
    key = name.strip().upper()
    if key not in _IPN_PRESETS:
        available = ", ".join(sorted(_IPN_PRESETS.keys()))
        raise ValueError(f"Unknown IPN profile '{name}'. Available: {available}")
    return _IPN_PRESETS[key]


def list_ipn() -> list[str]:
    return sorted(_IPN_PRESETS.keys())
