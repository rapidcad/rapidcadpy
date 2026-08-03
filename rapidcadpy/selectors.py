"""Selector helpers shared across backend integrations.

This module intentionally stays backend-agnostic. Backends only need to provide
edge direction vectors and can reuse the same selector semantics.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, TypeVar

EdgeT = TypeVar("EdgeT")


def parse_parallel_axis_selector(selector: Optional[str]) -> Optional[str]:
    """Parse a simple parallel-axis selector.

    Supported selectors mirror common CadQuery string syntax for directional
    edge filtering: ``|X``, ``|Y``, ``|Z``.
    """
    if selector is None:
        return None

    normalized = selector.strip().upper()
    if normalized in {"|X", "|Y", "|Z"}:
        return normalized[1]

    raise NotImplementedError(
        f"Unsupported edge selector '{selector}'. Only |X, |Y, and |Z are supported."
    )


def is_vector_parallel_to_axis(
    dx: float,
    dy: float,
    dz: float,
    axis: str,
    tolerance: float = 1e-3,
) -> bool:
    """Return True when the input vector is parallel to the requested axis."""
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length <= 1e-9:
        return False

    axis = axis.upper()
    axis_component = {
        "X": abs(dx) / length,
        "Y": abs(dy) / length,
        "Z": abs(dz) / length,
    }.get(axis)

    if axis_component is None:
        raise ValueError(f"Unsupported axis '{axis}'.")

    return axis_component >= 1.0 - tolerance


def filter_linear_edges_by_selector(
    edges: Iterable[EdgeT],
    selector: Optional[str],
    is_linear_edge: Callable[[EdgeT], bool],
    edge_direction_vector: Callable[[EdgeT], tuple[float, float, float]],
) -> List[EdgeT]:
    """Filter edges using a backend-agnostic selector implementation.

    This function contains the default string-selector behavior and is intended
    to be shared by all CAD kernels. Backends only provide adapter callbacks.
    """
    axis = parse_parallel_axis_selector(selector)
    edge_list = list(edges)

    if axis is None:
        return edge_list

    selected: List[EdgeT] = []
    for edge in edge_list:
        if not is_linear_edge(edge):
            continue

        dx, dy, dz = edge_direction_vector(edge)
        if is_vector_parallel_to_axis(dx, dy, dz, axis):
            selected.append(edge)

    return selected
