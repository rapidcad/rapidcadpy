"""Freeform intent validation without a CAD kernel or Pydantic dependency."""

import json

import pytest
from rapidcadpy.modeling import (
    CadPath,
    CadProfile,
    ControlPointSpline,
    CoordinateFrame,
    InterpolatedSpline,
    LoftDefinition,
    PathDefinition,
    ProfileDefinition,
    SweepDefinition,
    modeling_request_from_dict,
)

POINTS = ((0, 0, 0), (2, 3, 0), (4, 3, 0), (6, 0, 0))


@pytest.mark.parametrize(
    "definition",
    [
        ProfileDefinition(
            curves=(InterpolatedSpline(POINTS),),
            frame=CoordinateFrame(origin=(10, 20, 30), unit="in"),
            closed=False,
        ),
        PathDefinition(curves=(ControlPointSpline(POINTS, weights=(1, 2, 2, 1)),)),
        LoftDefinition(profiles=(CadProfile("doc", "p1"), CadProfile("doc", "p2"))),
        SweepDefinition(profile=CadProfile("doc", "p1"), path=CadPath("doc", "path")),
        LoftDefinition(
            profiles=(CadProfile("doc", "p1"), CadProfile("doc", "p2")),
            alignment="preserve",
        ),
    ],
)
def test_json_roundtrip_preserves_modeling_intent(definition):
    payload = json.loads(json.dumps(definition.to_dict(), allow_nan=False))
    assert modeling_request_from_dict(payload) == definition


def test_points_are_owned_and_interpolation_differs_from_control_points():
    mutable_points = [list(point) for point in POINTS]
    curve = InterpolatedSpline(mutable_points)
    mutable_points[1][0] = 100
    assert curve.points[1][0] == 2
    assert curve.to_dict()["kind"] != ControlPointSpline(POINTS).to_dict()["kind"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"degree": 0},
        {"degree": True},
        {"degree": 4},
        {"tolerance": 0},
        {"tolerance": float("nan")},
        {"points": ((0, 0, 0),) * 4},
        {"points": ((0, 0, 0), (0, 0, 0), (1, 0, 0), (2, 0, 0))},
        {"points": ((0, 0, 0), (1, 0, 0), (2, 0, 0), (float("inf"), 0, 0))},
    ],
)
def test_invalid_interpolation_is_rejected(kwargs):
    with pytest.raises(ValueError):
        InterpolatedSpline(**{"points": POINTS, **kwargs})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights": (1, 1)},
        {"weights": (1, 0, 1, 1)},
        {"weights": (1, float("inf"), 1, 1)},
        {"knots": (0, 0, 1, 1)},
        {"knots": (0, 0, 0, 0, 0, 0, 0, 0)},
        {"knots": (0, 0, 0, 0, 1, 0, 1, 1)},
        {"knots": (0, 0, 0, 0.5, 0.5, 1, 1, 1)},
        {"closed": True},
    ],
)
def test_invalid_control_point_spline_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ControlPointSpline(POINTS, **kwargs)


def test_expanded_clamped_knots_and_positional_closure():
    curve = ControlPointSpline(
        (*POINTS[:3], POINTS[0]), closed=True, knots=(0, 0, 0, 0, 1, 1, 1, 1)
    )
    assert curve.closed
    with pytest.raises(ValueError, match="Adjacent"):
        InterpolatedSpline((*POINTS[:3], POINTS[0]), closed=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unit": "feet"},
        {"normal": (0, 0, 0)},
        {"x_axis": (2, 0, 0)},
        {"normal": (1, 0, 0)},
        {"origin": (1, 2)},
    ],
)
def test_invalid_frame_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CoordinateFrame(**kwargs)


def test_profile_is_planar_but_path_can_be_spatial():
    curve = InterpolatedSpline((*POINTS[:3], (6, 0, 1)))
    assert PathDefinition((curve,)).curves == (curve,)
    with pytest.raises(ValueError, match="XY"):
        ProfileDefinition((curve,))


def test_persistent_references_are_document_scoped_and_ordered():
    first, second = CadProfile("doc", "p1"), CadProfile("doc", "p2")
    assert LoftDefinition((second, first)).profiles == (second, first)
    with pytest.raises(ValueError, match="distinct"):
        LoftDefinition((first, first))
    with pytest.raises(ValueError, match="same document"):
        LoftDefinition((first, CadProfile("other", "p2")))
    with pytest.raises(ValueError, match="same document"):
        SweepDefinition(first, CadPath("other", "path"))
    with pytest.raises(ValueError, match="orientation"):
        SweepDefinition(first, CadPath("doc", "path"), orientation="guess")


def test_payload_rejects_unknown_fields_and_tags():
    payload = ProfileDefinition((InterpolatedSpline(POINTS),)).to_dict()
    payload["curves"][0]["native_handle"] = "forbidden"
    with pytest.raises(TypeError):
        modeling_request_from_dict(payload)
    with pytest.raises(ValueError, match="Unknown"):
        modeling_request_from_dict({"kind": "arbitrary_code"})
