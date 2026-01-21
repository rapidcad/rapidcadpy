"""Public entry points for preset profile sections.

Intended usage:
    from rapidcadpy.components import profiles
    profile = profiles.ipe('IPE80').sketch(wp)
"""

from __future__ import annotations

from rapidcadpy.components.sections.ipe import ipe, list_ipe
from rapidcadpy.components.sections.ipn import ipn, list_ipn


def list_profiles() -> dict[str, list[str]]:
    """List available preset profile names by family."""

    return {
        "ipe": list_ipe(),
        "ipn": list_ipn(),
    }
