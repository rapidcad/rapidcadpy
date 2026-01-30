"""Public entry points for preset profile sections.

Intended usage:
    from .components import profiles
    profile = profiles.ipe('IPE80').sketch(wp)
"""

from __future__ import annotations

from .sections.ipe import list_ipe
from .sections.ipn import list_ipn


def list_profiles() -> dict[str, list[str]]:
    """List available preset profile names by family."""

    return {
        "ipe": list_ipe(),
        "ipn": list_ipn(),
    }
