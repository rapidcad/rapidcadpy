"""Portable feature intent and explicitly non-authoritative inferred features."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Optional
import uuid

from ..feature import Feature


Point3D = tuple[float, float, float]
Vector3D = tuple[float, float, float]


class FeatureProvenance(str, Enum):
    RAPIDCAD_CREATED = "rapidcad_created"
    NATIVE_HYDRATED = "native_hydrated"


class HoleTermination(str, Enum):
    THROUGH = "through"
    BLIND = "blind"


class HoleType(str, Enum):
    SIMPLE = "simple"
    COUNTERBORE = "counterbore"
    COUNTERSINK = "countersink"


class ChamferMode(str, Enum):
    EQUAL_DISTANCE = "equal_distance"
    TWO_DISTANCE = "two_distance"
    DISTANCE_ANGLE = "distance_angle"


@dataclass(frozen=True)
class GeometrySelection:
    """A revision-aware public subelement reference.

    ``geometry_signature`` is optional for callers selecting in the current
    revision, but executors must reject a changed revision without a usable,
    unambiguous signature.
    """

    object_id: str
    subelements: tuple[str, ...]
    document_revision: Optional[str] = None
    geometry_signature: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("GeometrySelection.object_id must not be empty.")
        if not self.subelements or any(not item for item in self.subelements):
            raise ValueError("GeometrySelection requires at least one subelement.")

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "subelements": list(self.subelements),
            "document_revision": self.document_revision,
            "geometry_signature": self.geometry_signature,
        }


@dataclass
class HoleFeature(Feature):
    """Authoritative intent for a subtractive hole operation."""

    target_id: str = ""
    center: Point3D = (0.0, 0.0, 0.0)
    axis: Vector3D = (0.0, 0.0, 1.0)
    diameter_mm: float = 1.0
    termination: HoleTermination = HoleTermination.THROUGH
    depth_mm: Optional[float] = None
    hole_type: HoleType = HoleType.SIMPLE
    countersink_diameter_mm: Optional[float] = None
    countersink_angle_degrees: Optional[float] = None
    provenance: FeatureProvenance = FeatureProvenance.RAPIDCAD_CREATED

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.target_id:
            raise ValueError("HoleFeature.target_id must not be empty.")
        if not math.isfinite(self.diameter_mm) or self.diameter_mm <= 0:
            raise ValueError("HoleFeature.diameter_mm must be positive.")
        if _vector_length(self.axis) <= 1.0e-9:
            raise ValueError("HoleFeature.axis must be non-zero.")
        if self.termination is HoleTermination.BLIND:
            if self.depth_mm is None or not math.isfinite(self.depth_mm) or self.depth_mm <= 0:
                raise ValueError("Blind HoleFeature requires a positive depth_mm.")
        elif self.depth_mm is not None:
            raise ValueError("Through HoleFeature must not declare depth_mm.")
        if self.hole_type is HoleType.COUNTERSINK:
            if self.countersink_diameter_mm is None or self.countersink_diameter_mm <= self.diameter_mm:
                raise ValueError("Countersunk HoleFeature requires countersink_diameter_mm larger than diameter_mm.")
            if self.countersink_angle_degrees is None or not 0.0 < self.countersink_angle_degrees < 180.0:
                raise ValueError("Countersunk HoleFeature requires countersink_angle_degrees between 0 and 180.")
        elif self.countersink_diameter_mm is not None or self.countersink_angle_degrees is not None:
            raise ValueError("Countersink dimensions require hole_type='countersink'.")

    def to_json(self) -> dict[str, Any]:
        return self.to_dict()

    def to_python(self, index: int = 0) -> str:
        return f"HoleFeature(target_id={self.target_id!r}, diameter_mm={self.diameter_mm!r})"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "hole",
            "id": str(self.id),
            "name": self.name,
            "target_id": self.target_id,
            "center": list(self.center),
            "axis": list(self.axis),
            "diameter_mm": self.diameter_mm,
            "termination": self.termination.value,
            "depth_mm": self.depth_mm,
            "hole_type": self.hole_type.value,
            "countersink_diameter_mm": self.countersink_diameter_mm,
            "countersink_angle_degrees": self.countersink_angle_degrees,
            "provenance": self.provenance.value,
        }


@dataclass
class FilletFeature(Feature):
    """Authoritative radius blend applied to one or more selected edges."""

    target_id: str = ""
    edges: tuple[GeometrySelection, ...] = ()
    radius_mm: float = 1.0
    reference_point: Optional[Point3D] = None
    provenance: FeatureProvenance = FeatureProvenance.RAPIDCAD_CREATED

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.target_id:
            raise ValueError("FilletFeature.target_id must not be empty.")
        if not self.edges:
            raise ValueError("FilletFeature requires at least one edge selection.")
        if any(selection.object_id != self.target_id for selection in self.edges):
            raise ValueError("FilletFeature selections must target target_id.")
        if not math.isfinite(self.radius_mm) or self.radius_mm <= 0:
            raise ValueError("FilletFeature.radius_mm must be positive.")

    def to_json(self) -> dict[str, Any]:
        return self.to_dict()

    def to_python(self, index: int = 0) -> str:
        return f"FilletFeature(target_id={self.target_id!r}, radius_mm={self.radius_mm!r})"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "fillet",
            "id": str(self.id),
            "name": self.name,
            "target_id": self.target_id,
            "edges": [edge.to_dict() for edge in self.edges],
            "radius_mm": self.radius_mm,
            "reference_point": list(self.reference_point) if self.reference_point else None,
            "provenance": self.provenance.value,
        }


@dataclass
class ChamferFeature(Feature):
    """Authoritative edge chamfer intent."""

    target_id: str = ""
    edges: tuple[GeometrySelection, ...] = ()
    distance_mm: float = 1.0
    mode: ChamferMode = ChamferMode.EQUAL_DISTANCE
    second_distance_mm: Optional[float] = None
    angle_degrees: Optional[float] = None
    reference_point: Optional[Point3D] = None
    provenance: FeatureProvenance = FeatureProvenance.RAPIDCAD_CREATED

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.target_id or not self.edges:
            raise ValueError("ChamferFeature requires target_id and edge selections.")
        if any(selection.object_id != self.target_id for selection in self.edges):
            raise ValueError("ChamferFeature selections must target target_id.")
        if not math.isfinite(self.distance_mm) or self.distance_mm <= 0:
            raise ValueError("ChamferFeature.distance_mm must be positive.")

    def to_json(self) -> dict[str, Any]:
        return self.to_dict()

    def to_python(self, index: int = 0) -> str:
        return f"ChamferFeature(target_id={self.target_id!r}, distance_mm={self.distance_mm!r})"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "chamfer",
            "id": str(self.id),
            "name": self.name,
            "target_id": self.target_id,
            "edges": [edge.to_dict() for edge in self.edges],
            "distance_mm": self.distance_mm,
            "mode": self.mode.value,
            "second_distance_mm": self.second_distance_mm,
            "angle_degrees": self.angle_degrees,
            "reference_point": list(self.reference_point) if self.reference_point else None,
            "provenance": self.provenance.value,
        }


def _vector_length(vector: Vector3D) -> float:
    return math.sqrt(sum(component * component for component in vector))


def feature_from_dict(value: dict[str, Any]) -> Feature:
    """Deserialize a persisted authoritative semantic definition."""

    kind = value.get("kind")
    common = {
        "id": uuid.UUID(str(value["id"])),
        "name": str(value.get("name", "Feature")),
        "target_id": str(value["target_id"]),
        "provenance": FeatureProvenance(value.get("provenance", "rapidcad_created")),
    }
    if kind == "hole":
        return HoleFeature(
            **common,
            center=tuple(value["center"]),  # type: ignore[arg-type]
            axis=tuple(value["axis"]),  # type: ignore[arg-type]
            diameter_mm=float(value["diameter_mm"]),
            termination=HoleTermination(value["termination"]),
            depth_mm=value.get("depth_mm"),
            hole_type=HoleType(value.get("hole_type", "simple")),
            countersink_diameter_mm=value.get("countersink_diameter_mm"),
            countersink_angle_degrees=value.get("countersink_angle_degrees"),
        )
    edge_selections = tuple(
        GeometrySelection(
            object_id=str(item["object_id"]),
            subelements=tuple(item["subelements"]),
            document_revision=item.get("document_revision"),
            geometry_signature=item.get("geometry_signature"),
        )
        for item in value.get("edges", [])
    )
    reference = value.get("reference_point")
    reference_point = tuple(reference) if reference else None
    if kind == "fillet":
        return FilletFeature(
            **common,
            edges=edge_selections,
            radius_mm=float(value["radius_mm"]),
            reference_point=reference_point,  # type: ignore[arg-type]
        )
    if kind == "chamfer":
        return ChamferFeature(
            **common,
            edges=edge_selections,
            distance_mm=float(value["distance_mm"]),
            mode=ChamferMode(value.get("mode", "equal_distance")),
            second_distance_mm=value.get("second_distance_mm"),
            angle_degrees=value.get("angle_degrees"),
            reference_point=reference_point,  # type: ignore[arg-type]
        )
    raise ValueError(f"Unsupported semantic feature kind {kind!r}.")
