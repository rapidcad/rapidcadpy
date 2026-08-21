"""Reusable ITEM fastening components.

These are assembly accessories, not section profiles. They carry the physical
and connection metadata that a CAD backend needs to place the corresponding
production part.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ItemAngleBracket:
    """Right-angle bracket for joining two ITEM Line 5 20x20 profiles."""

    name: str
    part_number: str
    leg_length: float
    width: float
    height: float
    mounting_hole_diameter: float
    mounting_hole_count: int
    material: str
    mass_g: float
    joint_angle_degrees: float = 90.0


_ITEM_ANGLE_BRACKETS: dict[str, ItemAngleBracket] = {
    "ITEM5_20X20_ANGLE_BRACKET": ItemAngleBracket(
        name="ITEM5_20X20_ANGLE_BRACKET",
        part_number="0.0.425.03",
        leg_length=20.0,
        width=20.0,
        height=20.0,
        mounting_hole_diameter=5.3,
        mounting_hole_count=2,
        material="die_cast_zinc",
        mass_g=14.0,
    ),
}


def item_angle_bracket(name: str) -> ItemAngleBracket:
    """Return an ITEM angle-bracket preset by name."""
    key = name.strip().upper().replace(" ", "")
    if key not in _ITEM_ANGLE_BRACKETS:
        available = ", ".join(sorted(_ITEM_ANGLE_BRACKETS))
        raise ValueError(f"Unknown ITEM angle bracket {name!r}. Available: {available}")
    return _ITEM_ANGLE_BRACKETS[key]


def list_item_angle_brackets() -> list[str]:
    """Return available ITEM angle-bracket preset names."""
    return sorted(_ITEM_ANGLE_BRACKETS)
