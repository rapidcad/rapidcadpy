"""Native, editable planar splines; no FreeCAD import until geometry is built."""

from __future__ import annotations

from itertools import groupby, pairwise
from typing import Any

from ...modeling import (
    ArcSegment,
    CircleSegment,
    ControlPointSpline,
    InterpolatedSpline,
    LineSegment,
    LoftDefinition,
    ModelingRequest,
    PathDefinition,
    ProfileDefinition,
    SweepDefinition,
    segment_points,
)
from ...operation_result import CadOperationSupport

UNIT_SCALE = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}
TRANSITIONS = {
    "transformed": "Transformed",
    "right": "Right corner",
    "round": "Round corner",
}


def definition_support(definition: ModelingRequest) -> CadOperationSupport:
    operation = {
        ProfileDefinition: "create_profile",
        PathDefinition: "create_path",
        LoftDefinition: "loft",
        SweepDefinition: "sweep",
    }.get(type(definition))
    if operation is None:
        raise TypeError("Expected a shared modeling definition.")
    reason = "Native linked features require valid references and geometry in the active document."
    status = "conditional"
    if isinstance(definition, LoftDefinition):
        reason = "Native loft retains section order and uses kernel matching. Closed single-edge sections require consistent winding and aligned seam directions."
        if definition.alignment != "automatic":
            status, reason = (
                "unsupported",
                "Part::Loft cannot disable kernel orientation/seam matching.",
            )
    elif isinstance(definition, (ProfileDefinition, PathDefinition)):
        reason = "Planar native sketches support lines, arcs, circles, cubic interpolation and clamped control-point splines of degree 1–25."
        for curve in definition.curves:
            points = segment_points(curve)
            if any(abs(p[2]) > curve.tolerance for p in points):
                status, reason = (
                    "unsupported",
                    "Spatial spline paths require a recomputable 3D curve representation; only local XY paths are supported.",
                )
                break
            if (isinstance(curve, InterpolatedSpline) and curve.degree != 3) or (
                isinstance(curve, ControlPointSpline) and curve.degree > 25
            ):
                status, reason = (
                    "unsupported",
                    "FreeCAD supports cubic interpolation only and control-point spline degrees 1–25.",
                )
                break
    elif isinstance(definition, SweepDefinition):
        reason = "Planar sketch spine; profile must lie at the start in a plane normal to its tangent. Frenet/corrected Frenet and transformed/right/round transitions map to native settings."
    return CadOperationSupport(
        operation,
        status,
        "native_dependent_feature"
        if operation in {"loft", "sweep"}
        else "native_feature",
        reason,
    )


def native_geometry(
    definition: ProfileDefinition | PathDefinition,
) -> tuple[list[Any], Any]:
    """Convert coordinates/tolerances to millimeters and retain the native frame."""
    import FreeCAD as App
    import Part

    support = definition_support(definition)
    if support.status == "unsupported":
        raise NotImplementedError(support.reason)
    scale = UNIT_SCALE[definition.frame.unit]
    geometries = []
    for spec in definition.curves:
        curve = Part.BSplineCurve()
        if isinstance(spec, InterpolatedSpline):
            points = [App.Vector(p[0] * scale, p[1] * scale, 0) for p in spec.points]
            if spec.closed:
                points.append(points[0])
            curve.interpolate(
                Points=points, PeriodicFlag=False, Tolerance=spec.tolerance * scale
            )
        elif isinstance(spec, ControlPointSpline):
            count, degree = len(spec.control_points), spec.degree
            expanded = spec.knots
            if expanded is None:
                interior = count - degree - 1
                expanded = (
                    (0.0,) * (degree + 1)
                    + tuple(i / (interior + 1) for i in range(1, interior + 1))
                    + (1.0,) * (degree + 1)
                )
            groups = [(k, len(list(values))) for k, values in groupby(expanded)]
            curve.buildFromPolesMultsKnots(
                [
                    App.Vector(p[0] * scale, p[1] * scale, 0)
                    for p in spec.control_points
                ],
                [n for _, n in groups],
                [k for k, _ in groups],
                False,
                degree,
                list(spec.weights or (1.0,) * count),
            )
        else:

            def vector(point):
                return App.Vector(point[0] * scale, point[1] * scale, 0)

            if isinstance(spec, LineSegment):
                curve = Part.LineSegment(vector(spec.start), vector(spec.end))
            elif isinstance(spec, ArcSegment):
                curve = Part.Arc(vector(spec.start), vector(spec.mid), vector(spec.end))
            else:
                curve = Part.Circle(
                    vector(spec.center), App.Vector(0, 0, 1), spec.radius * scale
                )
        if (
            isinstance(spec, (InterpolatedSpline, ControlPointSpline))
            and curve.Degree != spec.degree
        ):
            raise ValueError(
                f"Kernel produced degree {curve.Degree}; requested {spec.degree}."
            )
        closed = (
            spec.closed
            if isinstance(spec, (InterpolatedSpline, ControlPointSpline))
            else isinstance(spec, CircleSegment)
        )
        if curve.toShape().isClosed() != closed:
            raise ValueError(
                "Native spline closure differs from the requested closure."
            )
        geometries.append(curve)
    # Require ordered connectivity; never sort or repair a disconnected path.
    tolerance = max(c.tolerance for c in definition.curves) * scale
    for first, second in pairwise(geometries):
        first_edge, second_edge = first.toShape(), second.toShape()
        if (
            first_edge.valueAt(first_edge.LastParameter)
            - second_edge.valueAt(second_edge.FirstParameter)
        ).Length > tolerance:
            raise ValueError("Spline segments must connect in their specified order.")
    wire = Part.Wire([curve.toShape() for curve in geometries])
    if not wire.isValid():
        raise ValueError("Spline boundary is invalid.")
    if isinstance(definition, ProfileDefinition):
        if wire.isClosed() != definition.closed:
            raise ValueError("Profile boundary closure differs from requested closure.")
        if definition.closed and not Part.Face(wire).isValid():
            raise ValueError("Profile does not form a valid planar face.")
    frame = definition.frame
    normal, x_axis = App.Vector(*frame.normal), App.Vector(*frame.x_axis)
    rotation = App.Rotation(x_axis, normal.cross(x_axis), normal, "ZXY")
    placement = App.Placement(App.Vector(*(v * scale for v in frame.origin)), rotation)
    return geometries, placement


def validate_loft_sections(sections: list[Any]) -> None:
    """Reject ambiguous single-edge closed-wire correspondence before lofting.

    OCCT compatibility matching cannot reliably move a seam within one closed
    B-spline edge. Do not mutate source sketches to compensate for that limit.
    This rule also runs after dependency edits, before a transaction commits.
    """
    import FreeCAD as App
    import Part

    reference = None
    for section in sections:
        shape = section.Shape
        if shape.isNull() or len(shape.Wires) != 1:
            raise ValueError("Loft sections must contain exactly one wire.")
        wire = shape.Wires[0]
        if not wire.isClosed() or len(wire.Edges) != 1:
            continue
        normal = section.Placement.Rotation.multVec(App.Vector(0, 0, 1))
        samples = wire.discretize(Number=65)
        signed_area = sum(a.cross(b).dot(normal) for a, b in pairwise(samples))
        if abs(signed_area) < 1e-12:
            raise ValueError("Closed single-edge section has no unambiguous winding.")
        # Making a face attaches p-curves to its edges. Use a detached copy so
        # validation cannot alter the source sketch's B-Rep or its revision.
        centre = Part.Face(wire.copy()).CenterOfMass
        edge = wire.OrderedEdges[0]
        parameter = (
            edge.LastParameter
            if edge.Orientation == "Reversed"
            else edge.FirstParameter
        )
        seam = edge.valueAt(parameter) - centre
        if seam.Length < 1e-9:
            raise ValueError("Closed section seam direction cannot be determined.")
        seam.normalize()
        if reference is None:
            reference = normal, signed_area, seam
            continue
        first_normal, first_area, first_seam = reference
        if normal.dot(first_normal) < 1 - 1e-6:
            raise ValueError(
                "Closed single-edge loft sections require parallel, consistently oriented frames."
            )
        if signed_area * first_area <= 0:
            raise ValueError(
                "Closed single-edge loft sections require consistent winding; reverse the input curve explicitly."
            )
        if seam.dot(first_seam) < 1 - 1e-6:
            raise ValueError(
                "Closed single-edge loft seams must have aligned directions from their face centroids; choose corresponding start points explicitly."
            )


def sketch_wire(native: Any, *, closed: bool | None = None) -> Any:
    shape = native.Shape
    if (
        native.TypeId != "Sketcher::SketchObject"
        or shape.isNull()
        or len(shape.Wires) != 1
    ):
        raise ValueError(
            "Sections and spine must each be a single planar native sketch wire."
        )
    wire = shape.Wires[0]
    if len(wire.Edges) != len(shape.Edges) or not wire.isValid():
        raise ValueError("Sketch must contain exactly one connected, valid boundary.")
    if closed is not None and wire.isClosed() != closed:
        raise ValueError(
            "Sweep sections must be closed; the initial spine must be open."
        )
    return wire


def validate_sweep_inputs(profile: Any, path: Any) -> None:
    """Keep placement requirements true after edits of either dependency."""
    import FreeCAD as App

    sketch_wire(profile, closed=True)
    wire = sketch_wire(path, closed=False)
    edge = wire.OrderedEdges[0]
    parameter = (
        edge.LastParameter if edge.Orientation == "Reversed" else edge.FirstParameter
    )
    start, tangent = edge.valueAt(parameter), edge.tangentAt(parameter)
    normal = profile.Placement.Rotation.multVec(App.Vector(0, 0, 1))
    if (
        abs(abs(normal.dot(tangent)) - 1) > 1e-6
        or abs((start - profile.Placement.Base).dot(normal)) > 1e-6
    ):
        raise ValueError(
            "Place the profile at the spine start with its plane normal parallel to the initial tangent; automatic placement is unsupported."
        )


def modeling_capabilities() -> dict[str, Any]:
    operations = {}
    for name in (
        "create_profile",
        "update_profile",
        "create_path",
        "update_path",
        "loft",
        "sweep",
        "update_feature",
    ):
        operations[name] = CadOperationSupport(
            name,
            "conditional",
            "native_dependent_feature"
            if name in {"loft", "sweep", "update_feature"}
            else "native_feature",
            "Planar native sketch geometry and valid dependencies; constrained sketches cannot be replaced.",
        ).to_dict()
    return {
        "operations": operations,
        "restrictions": {
            "segments": [
                "line",
                "arc",
                "circle",
                "interpolated_spline",
                "control_point_spline",
            ],
            "interpolation_degrees": [3],
            "control_point_degrees": {"minimum": 1, "maximum": 25},
            "units": list(UNIT_SCALE),
            "path_dimensions": "planar_local_xy",
            "profile_loops": 1,
            "loft_alignment": ["automatic"],
            "loft_closed_single_edge": "parallel frames, consistent winding and aligned centroid-to-seam directions",
            "sweep_orientations": ["frenet", "corrected_frenet"],
            "sweep_transitions": list(TRANSITIONS),
            "sweep_sections": 1,
            "sweep_profile": "closed",
            "sweep_path": "open planar",
            "sweep_placement": "section plane passes through spine start, normal parallel to initial tangent",
            "feature_parameters": {
                "loft": ["profile_ids", "make_solid", "ruled", "alignment"],
                "sweep": [
                    "profile_id",
                    "path_id",
                    "make_solid",
                    "orientation",
                    "transition",
                ],
            },
        },
    }
