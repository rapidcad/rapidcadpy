"""Backend-neutral dimensions derived from authoritative semantic features."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .drawing import DimensionIntent
from .features import ChamferFeature, FilletFeature, HoleFeature, HoleTermination, HoleType


Point3D = tuple[float, float, float]


@dataclass(frozen=True)
class ModelBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def extents(self) -> tuple[float, float, float]:
        return (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        )


def build_dimension_intents_from_features(
    bounds: ModelBounds,
    features: Iterable[object],
) -> tuple[DimensionIntent, ...]:
    """Build a minimal dimension plan from known modelling intent.

    This deliberately does not infer intent from faces or edges. Adapters only
    need to supply bounds and canonical feature definitions.
    """

    feature_list = tuple(features)
    holes = tuple(item for item in feature_list if isinstance(item, HoleFeature))
    fillets = tuple(item for item in feature_list if isinstance(item, FilletFeature))
    chamfers = tuple(item for item in feature_list if isinstance(item, ChamferFeature))
    intents = _overall_intents(bounds)
    intents.extend(_hole_intents(holes))
    intents.extend(_fillet_intents(fillets, bounds))
    intents.extend(_chamfer_intents(chamfers, bounds))
    return tuple(intents)


def _overall_intents(bounds: ModelBounds) -> list[DimensionIntent]:
    return [
        DimensionIntent(
            id="overall:front:x", kind="overall", view="front",
            value_mm=bounds.x_max - bounds.x_min,
            reference_points=((bounds.x_min, bounds.y_min, bounds.z_max), (bounds.x_max, bounds.y_min, bounds.z_max)),
            preferred_sides=("top", "bottom"),
        ),
        DimensionIntent(
            id="overall:front:z", kind="overall", view="front",
            value_mm=bounds.z_max - bounds.z_min,
            reference_points=((bounds.x_min, bounds.y_min, bounds.z_min), (bounds.x_min, bounds.y_min, bounds.z_max)),
            preferred_sides=("left", "right"),
        ),
        DimensionIntent(
            id="overall:top:x", kind="overall", view="top",
            value_mm=bounds.x_max - bounds.x_min,
            reference_points=((bounds.x_min, bounds.y_min, bounds.z_max), (bounds.x_max, bounds.y_min, bounds.z_max)),
            preferred_sides=("bottom", "top"),
        ),
        DimensionIntent(
            id="overall:top:y", kind="overall", view="top",
            value_mm=bounds.y_max - bounds.y_min,
            reference_points=((bounds.x_min, bounds.y_min, bounds.z_max), (bounds.x_min, bounds.y_max, bounds.z_max)),
            preferred_sides=("left", "right"),
        ),
    ]


def _hole_intents(holes: Sequence[HoleFeature]) -> list[DimensionIntent]:
    groups: dict[tuple[float, tuple[float, float, float], str, float | None], list[HoleFeature]] = {}
    for hole in holes:
        axis = _normalized(hole.axis)
        key = (
            round(hole.diameter_mm, 6), axis, hole.termination.value,
            round(hole.depth_mm, 6) if hole.depth_mm is not None else None,
        )
        groups.setdefault(key, []).append(hole)
    intents: list[DimensionIntent] = []
    for index, group in enumerate(groups.values(), start=1):
        view = _axis_view(group[0].axis)
        representative = max(group, key=lambda item: _project(view, item.center)[0])
        radial = _view_u_axis(view)
        radius = representative.diameter_mm / 2.0
        intents.append(DimensionIntent(
            id=f"hole-group:{index}", kind="hole", view=view,
            value_mm=representative.diameter_mm,
            depth_mm=representative.depth_mm,
            through=representative.termination is HoleTermination.THROUGH,
            multiplicity=len(group),
            reference_points=(
                _add(representative.center, _scale(radial, -radius)),
                _add(representative.center, _scale(radial, radius)),
            ),
            preferred_sides=("right", "top", "left", "bottom"),
            source_references=tuple(str(item.id) for item in group),
        ))
        intents.extend(_hole_spacing_intents(group, index, view))
    for index, hole in enumerate(
        (item for item in holes if item.hole_type is HoleType.COUNTERSINK), start=1
    ):
        assert hole.countersink_diameter_mm is not None
        radius = hole.countersink_diameter_mm / 2.0
        radial = _view_u_axis(_axis_view(hole.axis))
        intents.append(DimensionIntent(
            id=f"countersink:{index}", kind="countersink", view=_axis_view(hole.axis),
            value_mm=hole.countersink_diameter_mm,
            angle_degrees=hole.countersink_angle_degrees,
            reference_points=(_add(hole.center, _scale(radial, -radius)), _add(hole.center, _scale(radial, radius))),
            preferred_sides=("right", "top", "left", "bottom"),
            source_references=(str(hole.id),),
        ))
    return intents


def _hole_spacing_intents(
    holes: Sequence[HoleFeature], group_index: int, view: str
) -> list[DimensionIntent]:
    if len(holes) < 2:
        return []
    coords = [(_project(view, hole.center), hole) for hole in holes]
    result: list[DimensionIntent] = []
    for axis_index, sides in ((0, ("bottom", "top")), (1, ("left", "right"))):
        fixed_axis = 1 - axis_index
        aligned_pairs = [
            (first, second)
            for index, first in enumerate(coords)
            for second in coords[index + 1 :]
            if math.isclose(
                first[0][fixed_axis], second[0][fixed_axis], abs_tol=1.0e-6
            )
            and not math.isclose(
                first[0][axis_index], second[0][axis_index], abs_tol=1.0e-6
            )
        ]
        distances = {
            round(abs(second[0][axis_index] - first[0][axis_index]), 6)
            for first, second in aligned_pairs
        }
        if len(distances) != 1 or not aligned_pairs:
            continue
        first, second = aligned_pairs[0][0][1], aligned_pairs[0][1][1]
        result.append(DimensionIntent(
            id=f"hole-group:{group_index}:spacing:{axis_index}", kind="spacing", view=view,
            value_mm=next(iter(distances)),
            reference_points=(first.center, second.center), preferred_sides=sides,
            source_references=tuple(str(item.id) for item in holes),
        ))
    return result


def _fillet_intents(features: Sequence[FilletFeature], bounds: ModelBounds) -> list[DimensionIntent]:
    groups: dict[float, list[FilletFeature]] = {}
    for feature in features:
        if feature.reference_point is not None:
            groups.setdefault(round(feature.radius_mm, 6), []).append(feature)
    result: list[DimensionIntent] = []
    for index, group in enumerate(groups.values(), start=1):
        feature = group[0]
        assert feature.reference_point is not None
        view = _best_feature_view(feature.reference_point, bounds)
        result.append(DimensionIntent(
            id=f"fillet-group:{index}", kind="radius", view=view,
            value_mm=feature.radius_mm, multiplicity=len(group),
            reference_points=(feature.reference_point,),
            preferred_sides=("right", "top", "left", "bottom"),
            source_references=tuple(str(item.id) for item in group),
        ))
    return result


def _chamfer_intents(features: Sequence[ChamferFeature], bounds: ModelBounds) -> list[DimensionIntent]:
    result: list[DimensionIntent] = []
    for index, feature in enumerate(features, start=1):
        if feature.reference_point is None:
            continue
        result.append(DimensionIntent(
            id=f"chamfer:{index}", kind="chamfer",
            view=_best_feature_view(feature.reference_point, bounds),
            value_mm=feature.distance_mm, angle_degrees=feature.angle_degrees,
            multiplicity=len(feature.edges), reference_points=(feature.reference_point,),
            preferred_sides=("right", "top", "left", "bottom"),
            source_references=(str(feature.id),),
        ))
    return result


def _axis_view(axis: Point3D) -> str:
    x, y, z = map(abs, _normalized(axis))
    return "top" if z >= max(x, y) else "front" if y >= x else "right"


def _best_feature_view(point: Point3D, bounds: ModelBounds) -> str:
    distances = {"front": min(abs(point[1] - bounds.y_min), abs(point[1] - bounds.y_max)), "top": min(abs(point[2] - bounds.z_min), abs(point[2] - bounds.z_max)), "right": min(abs(point[0] - bounds.x_min), abs(point[0] - bounds.x_max))}
    return max(distances, key=distances.get)


def _project(view: str, point: Point3D) -> tuple[float, float]:
    x, y, z = point
    return (x, z) if view == "front" else (x, y) if view == "top" else (y, z)


def _view_u_axis(view: str) -> Point3D:
    return (1.0, 0.0, 0.0) if view in {"front", "top"} else (0.0, 1.0, 0.0)


def _normalized(vector: Point3D) -> Point3D:
    length = math.sqrt(sum(value * value for value in vector))
    return tuple(round(value / length, 6) for value in vector)  # type: ignore[return-value]


def _add(first: Point3D, second: Point3D) -> Point3D:
    return tuple(a + b for a, b in zip(first, second))  # type: ignore[return-value]


def _scale(vector: Point3D, scalar: float) -> Point3D:
    return tuple(value * scalar for value in vector)  # type: ignore[return-value]
