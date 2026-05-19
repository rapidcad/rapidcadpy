"""Backend-neutral CAD history records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

from .primitives import Arc, Circle, Line

Primitive2D = Union[Line, Circle, Arc]


@dataclass(frozen=True)
class WorkplaneSpec:
    id: str
    name: str
    origin: Tuple[float, float, float]
    x_axis: Tuple[float, float, float]
    y_axis: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    offset: float = 0.0


@dataclass(frozen=True)
class SketchSpec:
    id: str
    workplane_id: str
    loops: Tuple[Tuple[Primitive2D, ...], ...]


@dataclass(frozen=True)
class ExtrudeSpec:
    id: str
    sketch_id: str
    distance: float
    operation: str = "NewBodyFeatureOperation"
    symmetric: bool = False


@dataclass(frozen=True)
class BooleanSpec:
    id: str
    kind: str
    base_feature_id: str
    tool_feature_ids: Tuple[str, ...]


@dataclass(frozen=True)
class FilletSpec:
    id: str
    base_feature_id: str
    radius: float
    edge_selector: str | None = None
