"""Serializable freeform modeling intent, independent of a CAD runtime.

Coordinates are local to their containing profile/path frame. Lengths (including
tolerances and frame origin) use that frame's unit; axes and weights are unitless.
Closed curves request positional closure only, not periodicity or smoothness.
Kernel-dependent feasibility is checked by the selected modeling backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import pairwise
from math import isfinite, sqrt
from typing import Any, Literal, TypeAlias

Point3: TypeAlias = tuple[float, float, float]
LengthUnit: TypeAlias = Literal["mm", "cm", "m", "in"]


def _point(value: Point3, name: str) -> Point3:
    if len(value) != 3 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
        for v in value
    ):
        raise ValueError(f"{name} must contain three finite numbers.")
    return tuple(float(v) for v in value)


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number.")
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


class ModelingDefinition:
    """JSON-compatible payload suitable for mapping from agent tool schemas."""

    def to_dict(self) -> dict[str, Any]:
        # JSON arrays, including nested point arrays, rather than native handles.
        def arrays(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: arrays(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [arrays(item) for item in value]
            return value

        return arrays(asdict(self))


@dataclass(frozen=True)
class CoordinateFrame(ModelingDefinition):
    """Right-handed local frame: Y = normal cross X; origin is in world space."""

    origin: Point3 = (0.0, 0.0, 0.0)
    x_axis: Point3 = (1.0, 0.0, 0.0)
    normal: Point3 = (0.0, 0.0, 1.0)
    unit: LengthUnit = "mm"

    def __post_init__(self) -> None:
        if self.unit not in {"mm", "cm", "m", "in"}:
            raise ValueError("Unsupported length unit.")
        for name in ("origin", "x_axis", "normal"):
            object.__setattr__(self, name, _point(getattr(self, name), name))
        for name in ("x_axis", "normal"):
            axis = getattr(self, name)
            length = sqrt(sum(v * v for v in axis))
            if abs(length - 1.0) > 1e-9:
                raise ValueError(f"{name} must be a unit vector.")
        if abs(sum(a * b for a, b in zip(self.x_axis, self.normal))) > 1e-9:
            raise ValueError("x_axis and normal must be perpendicular.")


def _validate_spline(curve: Any, points_name: str) -> None:
    points = tuple(_point(p, points_name) for p in getattr(curve, points_name))
    object.__setattr__(curve, points_name, points)
    if type(curve.degree) is not int or curve.degree < 1:
        raise ValueError("degree must be a positive integer.")
    if len(points) < curve.degree + 1:
        raise ValueError("A spline requires at least degree + 1 points.")
    if type(curve.closed) is not bool:
        raise ValueError("closed must be a boolean.")
    _positive(curve.tolerance, "tolerance")
    if all(p == points[0] for p in points):
        raise ValueError("A spline must contain distinct points.")


@dataclass(frozen=True)
class InterpolatedSpline(ModelingDefinition):
    """Pass through the points in order within tolerance.

    For a closed curve, omit the repeated endpoint; the backend closes it.
    Degree is requested exactly; an adapter must reject unsupported degrees.
    """

    points: tuple[Point3, ...]
    degree: int = 3
    closed: bool = False
    tolerance: float = 1e-6
    kind: Literal["interpolated_spline"] = field(
        default="interpolated_spline", init=False
    )

    def __post_init__(self) -> None:
        _validate_spline(self, "points")
        pairs = list(zip(self.points, self.points[1:]))
        if self.closed:
            pairs.append((self.points[-1], self.points[0]))
        if any(
            sum((a - b) ** 2 for a, b in zip(p, q)) <= self.tolerance**2
            for p, q in pairs
        ):
            raise ValueError("Adjacent interpolation points must exceed tolerance.")


@dataclass(frozen=True)
class ControlPointSpline(ModelingDefinition):
    """Clamped B-spline/NURBS defined by poles, not interpolation points.

    Knots are the expanded nondecreasing vector (multiplicities repeated).
    Omitted knots mean a uniform clamped vector; omitted weights mean all ones.
    A closed curve requires coincident first/last poles, with C0 closure only.
    Periodic splines are deliberately outside this initial contract.
    """

    control_points: tuple[Point3, ...]
    degree: int = 3
    closed: bool = False
    tolerance: float = 1e-6
    weights: tuple[float, ...] | None = None
    knots: tuple[float, ...] | None = None
    kind: Literal["control_point_spline"] = field(
        default="control_point_spline", init=False
    )

    def __post_init__(self) -> None:
        _validate_spline(self, "control_points")
        count = len(self.control_points)
        if self.weights is not None:
            object.__setattr__(self, "weights", tuple(self.weights))
            if len(self.weights) != count:
                raise ValueError("weights must match the control point count.")
            for weight in self.weights:
                _positive(weight, "weight")
        if self.knots is not None:
            knots = tuple(self.knots)
            object.__setattr__(self, "knots", knots)
            if len(knots) != count + self.degree + 1:
                raise ValueError("Expanded knot count must be pole count + degree + 1.")
            if any(
                isinstance(k, bool)
                or not isinstance(k, (int, float))
                or not isfinite(k)
                for k in knots
            ):
                raise ValueError("Knots must be finite numbers.")
            if any(a > b for a, b in pairwise(knots)) or knots[0] >= knots[-1]:
                raise ValueError("Knots must be nondecreasing with a nonzero domain.")
            multiplicity = self.degree + 1
            if (
                knots.count(knots[0]) != multiplicity
                or knots.count(knots[-1]) != multiplicity
            ):
                raise ValueError(
                    "End knots must have degree + 1 multiplicity (clamped)."
                )
            if any(
                knots.count(k) > self.degree
                for k in set(knots[multiplicity:-multiplicity])
            ):
                raise ValueError("Interior knot multiplicity cannot exceed degree.")
        if self.closed:
            first, last = self.control_points[0], self.control_points[-1]
            if sum((a - b) ** 2 for a, b in zip(first, last)) > self.tolerance**2:
                raise ValueError("Closed clamped splines require coincident end poles.")


SplineCurve: TypeAlias = InterpolatedSpline | ControlPointSpline


@dataclass(frozen=True)
class LineSegment(ModelingDefinition):
    start: Point3
    end: Point3
    tolerance: float = 1e-6
    kind: Literal["line"] = field(default="line", init=False)

    def __post_init__(self) -> None:
        for name in ("start", "end"):
            object.__setattr__(self, name, _point(getattr(self, name), name))
        _positive(self.tolerance, "tolerance")
        if sum((a - b) ** 2 for a, b in zip(self.start, self.end)) <= self.tolerance**2:
            raise ValueError("Line endpoints must exceed tolerance.")


@dataclass(frozen=True)
class ArcSegment(ModelingDefinition):
    start: Point3
    mid: Point3
    end: Point3
    tolerance: float = 1e-6
    kind: Literal["arc"] = field(default="arc", init=False)

    def __post_init__(self) -> None:
        for name in ("start", "mid", "end"):
            object.__setattr__(self, name, _point(getattr(self, name), name))
        _positive(self.tolerance, "tolerance")
        a = tuple(m - s for m, s in zip(self.mid, self.start))
        b = tuple(e - s for e, s in zip(self.end, self.start))
        cross = (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
        if sum(v * v for v in cross) <= self.tolerance**4:
            raise ValueError("Arc points must be distinct and non-collinear.")


@dataclass(frozen=True)
class CircleSegment(ModelingDefinition):
    center: Point3
    radius: float
    tolerance: float = 1e-6
    kind: Literal["circle"] = field(default="circle", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _point(self.center, "center"))
        _positive(self.radius, "radius")
        _positive(self.tolerance, "tolerance")


GeometrySegment: TypeAlias = SplineCurve | LineSegment | ArcSegment | CircleSegment


def segment_points(segment: GeometrySegment) -> tuple[Point3, ...]:
    if isinstance(segment, InterpolatedSpline):
        return segment.points
    if isinstance(segment, ControlPointSpline):
        return segment.control_points
    if isinstance(segment, LineSegment):
        return segment.start, segment.end
    if isinstance(segment, ArcSegment):
        return segment.start, segment.mid, segment.end
    return (segment.center,)


def _curves(value: tuple[GeometrySegment, ...]) -> tuple[GeometrySegment, ...]:
    curves = tuple(value)
    if not curves or any(
        not isinstance(
            c,
            (
                InterpolatedSpline,
                ControlPointSpline,
                LineSegment,
                ArcSegment,
                CircleSegment,
            ),
        )
        for c in curves
    ):
        raise ValueError("At least one typed spline curve is required.")
    return curves


@dataclass(frozen=True)
class ProfileDefinition(ModelingDefinition):
    """Ordered planar curves for a persistent native profile.

    Connectivity, closure and self-intersection require backend validation.
    Multiple loops/holes are deferred; curves describe one ordered boundary.
    """

    curves: tuple[GeometrySegment, ...]
    frame: CoordinateFrame = field(default_factory=CoordinateFrame)
    closed: bool = True
    name: str = "Profile"
    kind: Literal["profile"] = field(default="profile", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "curves", _curves(self.curves))
        if not isinstance(self.frame, CoordinateFrame):
            raise TypeError("frame must be a CoordinateFrame.")
        if type(self.closed) is not bool:
            raise ValueError("closed must be a boolean.")
        _identifier(self.name, "name")
        for curve in self.curves:
            points = segment_points(curve)
            if any(abs(point[2]) > curve.tolerance for point in points):
                raise ValueError("Profile points must lie in the local XY plane.")


@dataclass(frozen=True)
class PathDefinition(ModelingDefinition):
    """Ordered curves in a local frame; points may be spatial (non-planar)."""

    curves: tuple[GeometrySegment, ...]
    frame: CoordinateFrame = field(default_factory=CoordinateFrame)
    name: str = "Path"
    kind: Literal["path"] = field(default="path", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "curves", _curves(self.curves))
        if not isinstance(self.frame, CoordinateFrame):
            raise TypeError("frame must be a CoordinateFrame.")
        _identifier(self.name, "name")


@dataclass(frozen=True)
class CadProfile(ModelingDefinition):
    """Serializable persistent profile reference; never a workplane or handle."""

    document_id: str
    object_id: str

    def __post_init__(self) -> None:
        _identifier(self.document_id, "document_id")
        _identifier(self.object_id, "object_id")


@dataclass(frozen=True)
class CadPath(ModelingDefinition):
    """Serializable persistent path reference."""

    document_id: str
    object_id: str

    def __post_init__(self) -> None:
        _identifier(self.document_id, "document_id")
        _identifier(self.object_id, "object_id")


@dataclass(frozen=True)
class LoftDefinition(ModelingDefinition):
    """Ordered sections; automatic alignment may match winding and move seams.

    Preserve requests exact native wire correspondence. Backends unable to
    disable automatic matching must reject it rather than ignore it.
    """

    profiles: tuple[CadProfile, ...]
    make_solid: bool = True
    ruled: bool = False
    alignment: Literal["automatic", "preserve"] = "automatic"
    kind: Literal["loft"] = field(default="loft", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", tuple(self.profiles))
        if len(self.profiles) < 2 or any(
            not isinstance(p, CadProfile) for p in self.profiles
        ):
            raise ValueError("Loft requires at least two CadProfile references.")
        if len({p.document_id for p in self.profiles}) != 1:
            raise ValueError("Loft profiles must belong to the same document.")
        if len({p.object_id for p in self.profiles}) != len(self.profiles):
            raise ValueError("Loft profiles must be distinct.")
        if type(self.make_solid) is not bool or type(self.ruled) is not bool:
            raise ValueError("make_solid and ruled must be booleans.")
        if self.alignment not in {"automatic", "preserve"}:
            raise ValueError("Unknown loft alignment.")


@dataclass(frozen=True)
class SweepDefinition(ModelingDefinition):
    """Create a dependent sweep; unsupported frame/transition modes must fail."""

    profile: CadProfile
    path: CadPath
    make_solid: bool = True
    orientation: Literal["frenet", "corrected_frenet"] = "corrected_frenet"
    transition: Literal["transformed", "right", "round"] = "right"
    kind: Literal["sweep"] = field(default="sweep", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.profile, CadProfile) or not isinstance(
            self.path, CadPath
        ):
            raise TypeError("Sweep requires a CadProfile and CadPath.")
        if self.profile.document_id != self.path.document_id:
            raise ValueError("Sweep inputs must belong to the same document.")
        if self.orientation not in {"frenet", "corrected_frenet"}:
            raise ValueError("Unknown sweep orientation.")
        if self.transition not in {"transformed", "right", "round"}:
            raise ValueError("Unknown sweep transition.")
        if type(self.make_solid) is not bool:
            raise ValueError("make_solid must be a boolean.")


ModelingRequest: TypeAlias = (
    ProfileDefinition | PathDefinition | LoftDefinition | SweepDefinition
)


def modeling_request_from_dict(payload: dict[str, Any]) -> ModelingRequest:
    """Decode tagged JSON data; constructors reject unknown fields/invalid values.

    This is the mapping boundary for tool schemas and connector payloads. It
    requires no Pydantic dependency and never resolves native objects.
    """
    data = dict(payload)
    kind = data.pop("kind", None)
    if kind in {"profile", "path"}:
        curves = []
        for value in data.pop("curves"):
            curve_data = dict(value)
            curve_kind = curve_data.pop("kind", None)
            if curve_kind == "interpolated_spline":
                curves.append(InterpolatedSpline(**curve_data))
            elif curve_kind == "control_point_spline":
                curves.append(ControlPointSpline(**curve_data))
            elif curve_kind in {"line", "arc", "circle"}:
                constructor = {
                    "line": LineSegment,
                    "arc": ArcSegment,
                    "circle": CircleSegment,
                }[curve_kind]
                curves.append(constructor(**curve_data))
            else:
                raise ValueError(f"Unknown spline kind: {curve_kind!r}.")
        frame = CoordinateFrame(**data.pop("frame", {}))
        constructor = ProfileDefinition if kind == "profile" else PathDefinition
        return constructor(curves=tuple(curves), frame=frame, **data)
    if kind == "loft":
        profiles = tuple(CadProfile(**item) for item in data.pop("profiles"))
        return LoftDefinition(profiles=profiles, **data)
    if kind == "sweep":
        profile = CadProfile(**data.pop("profile"))
        path = CadPath(**data.pop("path"))
        return SweepDefinition(profile=profile, path=path, **data)
    raise ValueError(f"Unknown modeling request kind: {kind!r}.")
