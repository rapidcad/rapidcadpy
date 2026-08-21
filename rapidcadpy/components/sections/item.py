"""ITEM aluminium extrusion section presets.

The initial preset models the ITEM 5, 20x20 profile from the supplied section
drawing. It preserves the outer envelope, four T-slot openings, and central
bore. The drawing's R2 transitions are represented by a faceted outline until
the common Workplane API has backend-neutral sketch fillets.

Usage:
    from .components import profiles
    beam = profiles.item("ITEM5_20X20").sketch(wp).extrude(300)
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Section2D


@dataclass(frozen=True)
class ItemSection(Section2D):
    """Parametric ITEM-compatible T-slot extrusion profile."""

    name: str
    width: float
    height: float
    slot_opening: float
    slot_depth: float
    wall_thickness: float
    core_hole_diameter: float
    corner_radius: float

    def sketch(self, wp, *, x: float = 0.0, y: float = 0.0):
        """Sketch the faceted profile centered on ``(x, y)``.

        The outer loop has one open T-slot on each face. The central circular
        loop is the profile's through-bore, so supported CAD backends extrude it
        as a void.
        """
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ITEM section width and height must be positive.")
        if self.slot_opening <= 0 or self.slot_depth <= 0:
            raise ValueError("ITEM slot dimensions must be positive.")
        if self.core_hole_diameter <= 0:
            raise ValueError("ITEM core hole diameter must be positive.")

        half_width = self.width / 2.0
        half_height = self.height / 2.0
        half_slot = self.slot_opening / 2.0
        # Four notches form the shared, backend-neutral approximation of the
        # T-slots. Keep their roots inside the central material core.
        root_x = half_width - self.slot_depth
        root_y = half_height - self.slot_depth

        points = [
            (x - half_width, y + half_height),
            (x - half_slot, y + half_height),
            (x - half_slot, y + root_y),
            (x + half_slot, y + root_y),
            (x + half_slot, y + half_height),
            (x + half_width, y + half_height),
            (x + half_width, y + half_slot),
            (x + root_x, y + half_slot),
            (x + root_x, y - half_slot),
            (x + half_width, y - half_slot),
            (x + half_width, y - half_height),
            (x + half_slot, y - half_height),
            (x + half_slot, y - root_y),
            (x - half_slot, y - root_y),
            (x - half_slot, y - half_height),
            (x - half_width, y - half_height),
            (x - half_width, y - half_slot),
            (x - root_x, y - half_slot),
            (x - root_x, y + half_slot),
            (x - half_width, y + half_slot),
            (x - half_width, y + half_height),
        ]
        wp.move_to(*points[0])
        for point in points[1:]:
            wp.line_to(*point)

        # The outer outline is explicitly closed before starting the inner
        # bore loop. This avoids a connector line in Workplane.close().
        wp.move_to(x, y).circle(self.core_hole_diameter / 2.0)
        return wp.close()


_ITEM_PRESETS: dict[str, ItemSection] = {
    "ITEM5_20X20": ItemSection(
        name="ITEM5_20X20",
        width=20.0,
        height=20.0,
        slot_opening=6.35,
        slot_depth=5.0,
        wall_thickness=1.8,
        core_hole_diameter=4.3,
        corner_radius=2.0,
    ),
}


def item(name: str) -> ItemSection:
    """Return an ITEM profile preset by name."""
    key = name.strip().upper().replace(" ", "")
    if key not in _ITEM_PRESETS:
        available = ", ".join(sorted(_ITEM_PRESETS))
        raise ValueError(f"Unknown ITEM profile {name!r}. Available: {available}")
    return _ITEM_PRESETS[key]


def list_item() -> list[str]:
    """Return available ITEM profile preset names."""
    return sorted(_ITEM_PRESETS)
