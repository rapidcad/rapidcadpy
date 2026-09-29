"""Run real FreeCAD assertions in its own interpreter; invoked by pytest."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import FreeCAD as App
import Part
import Sketcher
from rapidcadpy import (
    CadPath,
    CadProfile,
    CadSession,
    ControlPointSpline,
    CoordinateFrame,
    InterpolatedSpline,
    LoftDefinition,
    PathDefinition,
    ProfileDefinition,
    SweepDefinition,
)
from rapidcadpy.integrations.freecad.app import FreeCADApp
from rapidcadpy.integrations.freecad.mutations import FreeCADMutationBackend
from rapidcadpy.integrations.freecad.sketch2d import FreeCADSketch2D


def check(result):
    assert result["ok"], result
    return result


def setup():
    session = CadSession(execution_mode="embedded")
    check(session.new_document("ProfileTest"))
    first = check(session.work_plane("XY"))["object_id"]
    check(session.circle(10))
    second = check(session.work_plane("XY", offset=30))["object_id"]
    check(session.circle(15))
    return session, first, second


def reuse_and_classification(directory):
    session, first, second = setup()
    handles = [session.runtime_objects[first], session.runtime_objects[second]]
    before = [copy.deepcopy(wp._pending_shapes) for wp in handles]
    solid = check(session.loft([first, second]))
    assert solid["object_type"] == "solid"
    assert solid["object_id"] in solid["created_object_ids"]
    assert set(solid["profile_ids"]) <= set(solid["created_object_ids"])
    assert len(session.profile_references) == 2
    revision = solid["document_revision"]
    session.cad_document.adapter.profile_operations(session)._refresh()
    assert session.document_revision == revision
    shell = check(
        session.loft(
            profile_ids=solid["profile_ids"],
            make_solid=False,
            expected_revision=revision,
        )
    )
    assert shell["object_type"] == "shell"
    assert shell["created_object_ids"] == [shell["object_id"]]
    again = check(
        session.loft([first, second], expected_revision=shell["document_revision"])
    )
    assert again["profile_ids"] == solid["profile_ids"]
    assert len(session.profile_references) == 2
    for wp, primitives in zip(handles, before):
        assert [(type(p).__name__, vars(p)) for p in wp._pending_shapes] == [
            (type(p).__name__, vars(p)) for p in primitives
        ]
    for result in (solid, shell):
        native = session.runtime_objects[result["object_id"]].native_handle
        assert native.TypeId == "Part::Loft"
        assert [obj.RapidCADObjectId for obj in native.Sections] == solid["profile_ids"]
    encoded = json.dumps(shell, allow_nan=False)
    assert "native_handle" not in encoded
    # The fluent convenience API also reuses profiles without consuming closed loops.
    app = FreeCADApp(doc_name="FluentProfileTest")
    a = app.work_plane("XY").circle(5)
    b = app.work_plane("XY", offset=10).circle(7)
    a.close()
    b.close()
    first_shape = a.loft(b)
    second_shape = a.loft(b, make_solid=False)
    assert (
        first_shape.feature.native_handle.Sections
        == second_shape.feature.native_handle.Sections
    )
    assert len(a._accumulated_loops) == len(b._accumulated_loops) == 1


def update_and_reopen(directory):
    session, first, second = setup()
    original = check(session.loft([first, second]))
    profile_id = original["profile_ids"][0]
    loft_id = original["object_id"]
    native_loft = session.runtime_objects[loft_id].native_handle
    profile_name = session.runtime_objects[profile_id].native_name
    volume = native_loft.Shape.Volume
    replacement = check(session.work_plane("XY"))["object_id"]
    check(session.circle(20))
    updated = check(
        session.update_profile(
            profile_id, replacement, expected_revision=original["document_revision"]
        )
    )
    assert profile_id in updated["changed_object_ids"]
    assert loft_id in updated["changed_object_ids"]
    assert not updated["created_object_ids"] and not updated["removed_object_ids"]
    assert native_loft.Sections[0].Name == profile_name
    assert native_loft.Shape.Volume != volume
    references = [session.profile_references[key] for key in original["profile_ids"]]
    path = Path(directory) / "persistent.FCStd"
    check(session.save_document(str(path)))
    App.closeDocument(session.cad_document.native_handle.Name)
    document = App.openDocument(str(path))
    reopened = CadSession(execution_mode="embedded")
    reopened.app = FreeCADApp.from_document(document)
    reopened.backend_name = "freecad"
    check(reopened.use_active_document())
    # FreeCAD normalizes attachment engines and B-Reps on reload. IDs persist,
    # while the freshly hydrated revision describes that loaded native state.
    reopened_revision = reopened.document_revision
    reopened.cad_document.adapter.profile_operations(reopened)._refresh()
    assert reopened.document_revision == reopened_revision
    assert list(reopened.profile_references.values()) == references
    assert reopened.cad_document.id == references[0].document_id
    assert loft_id in reopened.objects
    definition = LoftDefinition(tuple(references), make_solid=False)
    result = check(
        reopened.apply_modeling(
            definition.to_dict(), expected_revision=reopened.document_revision
        )
    )
    assert result["object_type"] == "shell"
    wrong = LoftDefinition(
        tuple(CadProfile("another-document", ref.object_id) for ref in references)
    )
    assert not reopened.apply_modeling(
        wrong, expected_revision=result["document_revision"]
    )["ok"]


def rollback_creation_and_hydration(directory):
    for stage in ("create", "hydrate", "validate"):
        session, first, second = setup()
        before = copy.deepcopy(session.objects)
        states = [
            (wp, copy.deepcopy(wp._pending_shapes))
            for wp in (session.runtime_objects[first], session.runtime_objects[second])
        ]
        native = session.cad_document.native_handle
        meta = dict(native.Meta)
        original_builder = FreeCADSketch2D._create_editable_sketch
        original_hydrate = session._hydrate_freecad_document
        original_validate = FreeCADMutationBackend.validate
        calls = 0

        def broken_builder(self, doc, name, original_builder=original_builder):
            original_builder(self, doc, name)
            raise RuntimeError("injected sketch failure")

        def broken_hydrate(*args, original_hydrate=original_hydrate, **kwargs):
            nonlocal calls
            calls += 1
            result = original_hydrate(*args, **kwargs)
            if calls == 2:
                raise RuntimeError("injected hydration failure")
            return result

        def broken_validate(self):
            raise ValueError("injected validation failure")

        try:
            if stage == "create":
                FreeCADSketch2D._create_editable_sketch = broken_builder
            if stage == "hydrate":
                session._hydrate_freecad_document = broken_hydrate
            if stage == "validate":
                FreeCADMutationBackend.validate = broken_validate
            result = session.loft([first, second])
        finally:
            FreeCADSketch2D._create_editable_sketch = original_builder
            session._hydrate_freecad_document = original_hydrate
            FreeCADMutationBackend.validate = original_validate
        assert not result["ok"], result
        assert result["error_code"] == "cad_operation_failed", result
        assert not native.Objects
        assert dict(native.Meta) == meta
        assert native.UndoMode == 0 and not native.HasPendingTransaction
        assert session.objects == before
        assert session.profile_references == session._profile_cache == {}
        for wp, primitives in states:
            assert [(type(p).__name__, vars(p)) for p in wp._pending_shapes] == [
                (type(p).__name__, vars(p)) for p in primitives
            ]
        check(session.loft([first, second]))


def rollback_update_and_external_revision(directory):
    session, first, second = setup()
    created = check(session.loft([first, second]))
    profile_id, loft_id = created["profile_ids"][0], created["object_id"]
    native_profile = session.runtime_objects[profile_id].native_handle
    native_loft = session.runtime_objects[loft_id].native_handle
    volume, radius = native_loft.Shape.Volume, native_profile.Geometry[0].Radius
    revision = created["document_revision"]
    replacement = check(session.work_plane("XY"))["object_id"]
    check(session.circle(25))
    before = copy.deepcopy(session.objects)
    original_validate = FreeCADMutationBackend.validate

    def broken_validate(self):
        raise RuntimeError("injected dependent feature failure")

    try:
        FreeCADMutationBackend.validate = broken_validate
        result = session.update_profile(
            profile_id, replacement, expected_revision=revision
        )
    finally:
        FreeCADMutationBackend.validate = original_validate
    assert not result["ok"], result
    assert native_profile.Geometry[0].Radius == radius
    assert abs(native_loft.Shape.Volume - volume) < 1e-6
    assert session.objects == before
    assert session.document_revision == revision
    check(session.update_profile(profile_id, replacement, expected_revision=revision))
    current = session.document_revision
    native_profile.delGeometry(0)
    native_profile.addGeometry(
        Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1), 30), False
    )
    session.cad_document.native_handle.recompute()
    count = len(session.cad_document.native_handle.Objects)
    stale = session.loft(profile_ids=created["profile_ids"], expected_revision=current)
    assert stale["error_code"] == "document_revision_mismatch", stale
    assert len(session.cad_document.native_handle.Objects) == count
    assert native_profile.Geometry[0].Radius == 30
    check(
        session.loft(
            profile_ids=created["profile_ids"],
            expected_revision=stale["document_revision"],
        )
    )


def constraints_paths_and_invalid_inputs(directory):
    session, first, second = setup()
    profile = check(session.create_profile(first))
    native = session.runtime_objects[profile["object_id"]].native_handle
    native.addConstraint(Sketcher.Constraint("Diameter", 0, 20))
    session.cad_document.native_handle.recompute()
    failed = session.update_profile(profile["object_id"], second)
    assert not failed["ok"] and native.ConstraintCount == 1
    check(session.use_active_document())
    support = session.get_object(profile["object_id"])["object"]["operation_support"][
        "update_profile"
    ]
    assert support["status"] == "unsupported"
    path_wp = check(session.work_plane("XY"))["object_id"]
    check(session.move_to(0, 0))
    check(session.line_to(10, 0))
    path = check(session.create_path(path_wp))
    assert path["path"]["object_id"] in session.path_references
    count = len(session.cad_document.native_handle.Objects)
    invalid = session.loft(profile_ids=[profile["object_id"], path["object_id"]])
    assert not invalid["ok"]
    assert len(session.cad_document.native_handle.Objects) == count
    # An open section cannot silently become a solid.
    open_profile = check(session.create_profile(path_wp))
    invalid = session.loft(
        profile_ids=[profile["object_id"], open_profile["object_id"]], make_solid=True
    )
    assert not invalid["ok"]
    assert len(session.cad_document.native_handle.Objects) == count + 1
    # Mixed closed loops and new pending geometry must not silently discard a loop.
    mixed_wp = check(session.work_plane("XY"))["object_id"]
    check(session.circle(3))
    session.runtime_objects[mixed_wp].close()
    check(session.circle(5))
    failed = session.create_profile(mixed_wp)
    assert not failed["ok"]
    assert len(session.cad_document.native_handle.Objects) == count + 1
    session.cad_document.native_handle.UndoMode = 1
    session.cad_document.native_handle.openTransaction("User edit")
    native.Label = "User changed label"
    failed = session.create_profile(path_wp)
    assert not failed["ok"]
    assert session.cad_document.native_handle.HasPendingTransaction
    assert native.Label == "User changed label"
    session.cad_document.native_handle.abortTransaction()


def spline_sweep_setup():
    session = CadSession(execution_mode="embedded")
    check(session.new_document("SplineSweeps"))
    definition = PathDefinition(
        (ControlPointSpline(((0, 0, 0), (0, 10, 0), (10, 20, 0), (10, 30, 0))),)
    )
    path = check(
        session.apply_modeling(definition, expected_revision=session.document_revision)
    )
    wp = check(session.work_plane("XZ"))["object_id"]
    check(session.circle(1))
    profile = check(
        session.create_profile(wp, expected_revision=path["document_revision"])
    )
    return session, profile["object_id"], path["object_id"], definition


def planar_spline_sweeps(directory):
    session, profile_id, path_id, _definition = spline_sweep_setup()
    path = session.runtime_objects[path_id].native_handle
    assert path.TypeId == "Sketcher::SketchObject" and path.Geometry[0].Degree == 3
    profile = session.runtime_objects[profile_id].native_handle
    for orientation in ("frenet", "corrected_frenet"):
        for transition, native_mode in (
            ("transformed", "Transformed"),
            ("right", "Right corner"),
            ("round", "Round corner"),
        ):
            for solid in (True, False):
                result = check(
                    session.sweep(
                        profile_id,
                        path_id,
                        make_solid=solid,
                        orientation=orientation,
                        transition=transition,
                        expected_revision=session.document_revision,
                    )
                )
                native = session.runtime_objects[result["object_id"]].native_handle
                assert native.TypeId == "Part::Sweep"
                assert native.Sections == [profile] and native.Spine == (path, [])
                assert native.Frenet == (orientation == "frenet")
                assert native.Transition == native_mode
                assert result["created_object_ids"] == [result["object_id"]]
                assert result["object_type"] == ("solid" if solid else "shell")
                assert native.Shape.isValid()
    # Replacing path geometry keeps the linked sketch and recomputes all sweeps.
    before_volumes = {
        obj.Name: obj.Shape.Volume
        for obj in session.cad_document.native_handle.Objects
        if obj.TypeId == "Part::Sweep"
    }
    replacement = PathDefinition(
        (ControlPointSpline(((0, 0, 0), (0, 12, 0), (15, 25, 0), (15, 40, 0))),)
    )
    updated = check(
        session.update_path(
            path_id, replacement, expected_revision=session.document_revision
        )
    )
    assert (
        updated["created_object_ids"] == [] and path_id in updated["changed_object_ids"]
    )
    for native in session.cad_document.native_handle.Objects:
        if native.TypeId == "Part::Sweep":
            assert native.Spine[0] is path
            if native.Solid:
                assert native.Shape.Volume != before_volumes[native.Name]
                assert native.RapidCADObjectId in updated["changed_object_ids"]
    wp = check(session.work_plane("XZ"))["object_id"]
    check(session.circle(2))
    previous = {
        obj.RapidCADObjectId: obj.Shape.Volume
        for obj in session.cad_document.native_handle.Objects
        if obj.TypeId == "Part::Sweep" and obj.Solid
    }
    resized = check(
        session.update_profile(
            profile_id, wp, expected_revision=session.document_revision
        )
    )
    assert profile_id in resized["changed_object_ids"]
    for object_id, volume in previous.items():
        native = session.runtime_objects[object_id].native_handle
        assert native.Shape.Volume > volume
        assert object_id in resized["changed_object_ids"]
    typed = SweepDefinition(
        session.profile_references[profile_id], session.path_references[path_id]
    )
    check(
        session.apply_modeling(
            typed.to_dict(), expected_revision=session.document_revision
        )
    )
    filename = Path(directory) / "sweeps.FCStd"
    check(session.save_document(str(filename)))
    App.closeDocument(session.cad_document.native_handle.Name)
    document = App.openDocument(str(filename))
    reopened = CadSession(execution_mode="embedded")
    reopened.app = FreeCADApp.from_document(document)
    reopened.backend_name = "freecad"
    check(reopened.use_active_document())
    assert reopened.path_references[path_id] == typed.path
    assert reopened.objects[updated["object_id"]].type == "path"
    sweeps = [obj for obj in document.Objects if obj.TypeId == "Part::Sweep"]
    assert len(sweeps) == 13 and all(
        obj.Spine[0].RapidCADObjectId == path_id for obj in sweeps
    )
    document.recompute()
    assert all(obj.Shape.isValid() for obj in sweeps)


def sweep_corner_modes(directory):
    session = CadSession(execution_mode="embedded")
    check(session.new_document("Corners"))
    path_wp = check(session.work_plane("XY"))["object_id"]
    check(session.move_to(0, 0))
    check(session.line_to(0, 10))
    check(session.line_to(10, 10))
    path = check(session.create_path(path_wp))["object_id"]
    profile_wp = check(session.work_plane("XZ"))["object_id"]
    check(session.circle(1))
    profile = check(session.create_profile(profile_wp))["object_id"]
    for transition in ("right", "round"):
        result = check(
            session.sweep(
                profile,
                path,
                transition=transition,
                expected_revision=session.document_revision,
            )
        )
        assert result["object_type"] == "solid"
        native = session.runtime_objects[result["object_id"]].native_handle
        assert native.Shape.isValid() and native.Shape.Volume > 0


def spline_inputs_and_rejections(directory):
    session = CadSession(execution_mode="embedded")
    check(session.new_document("SplineInputs"))
    spec = InterpolatedSpline(((0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)))
    request = PathDefinition(
        (spec,), frame=CoordinateFrame(origin=(1, 2, 3), unit="cm")
    )
    result = check(
        session.apply_modeling(request, expected_revision=session.document_revision)
    )
    native = session.runtime_objects[result["object_id"]].native_handle
    assert native.Geometry[0].Degree == 3
    assert native.Shape.Vertexes[0].Point.isEqual(App.Vector(10, 20, 30), 1e-6)
    assert abs(native.Shape.Length - 30) < 1e-6
    weighted = ControlPointSpline(
        ((0, 0, 0), (0, 10, 0), (10, 20, 0), (10, 30, 0)),
        weights=(1, 2, 2, 1),
        knots=(0, 0, 0, 0, 1, 1, 1, 1),
    )
    result = check(
        session.apply_modeling(
            PathDefinition((weighted,)), expected_revision=session.document_revision
        )
    )
    native = session.runtime_objects[result["object_id"]].native_handle
    assert native.Geometry[0].isRational() and native.Geometry[0].getWeights() == [
        1,
        2,
        2,
        1,
    ]
    count = len(session.cad_document.native_handle.Objects)
    unsupported = [
        PathDefinition(
            (InterpolatedSpline(((0, 0, 0), (1, 1, 0), (2, 2, 0)), degree=2),)
        ),
        PathDefinition(
            (ControlPointSpline(((0, 0, 0), (0, 10, 1), (10, 20, 0), (10, 30, 0))),)
        ),
    ]
    for request in unsupported:
        assert (
            session.inspect_modeling_support(request)["support"]["status"]
            == "unsupported"
        )
        rejected = session.apply_modeling(
            request, expected_revision=session.document_revision
        )
        assert rejected["error_code"] == "cad_operation_not_supported"
        assert len(session.cad_document.native_handle.Objects) == count
    disconnected = PathDefinition(
        (spec, InterpolatedSpline(((10, 0, 0), (10, 1, 0), (10, 2, 0), (10, 3, 0))))
    )
    assert not session.apply_modeling(
        disconnected, expected_revision=session.document_revision
    )["ok"]
    assert len(session.cad_document.native_handle.Objects) == count


def sweep_rollback_and_invalid_modes(directory):
    session, profile_id, path_id, _definition = spline_sweep_setup()
    count = len(session.cad_document.native_handle.Objects)
    for params in ({"orientation": "fixed"}, {"transition": "magic"}):
        assert not session.sweep(profile_id, path_id, **params)["ok"]
        assert len(session.cad_document.native_handle.Objects) == count
    assert not session.sweep(path_id, profile_id)["ok"]
    wrong_wp = check(session.work_plane("XY"))["object_id"]
    check(session.circle(1))
    wrong_profile = check(session.create_profile(wrong_wp))["object_id"]
    assert not session.sweep(wrong_profile, path_id)["ok"]
    before = copy.deepcopy(session.objects)
    original_validate = FreeCADMutationBackend.validate

    def broken_validate(self):
        raise RuntimeError("injected sweep failure")

    try:
        FreeCADMutationBackend.validate = broken_validate
        assert not session.sweep(profile_id, path_id)["ok"]
    finally:
        FreeCADMutationBackend.validate = original_validate
    assert session.objects == before
    assert not any(
        obj.TypeId == "Part::Sweep"
        for obj in session.cad_document.native_handle.Objects
    )
    check(session.sweep(profile_id, path_id))
    before = copy.deepcopy(session.objects)
    path = session.runtime_objects[path_id].native_handle
    length = path.Shape.Length
    revision = session.document_revision
    try:
        FreeCADMutationBackend.validate = broken_validate
        replacement = PathDefinition(
            (ControlPointSpline(((0, 0, 0), (0, 15, 0), (15, 20, 0), (15, 50, 0))),)
        )
        assert not session.update_path(
            path_id, replacement, expected_revision=revision
        )["ok"]
    finally:
        FreeCADMutationBackend.validate = original_validate
    assert abs(path.Shape.Length - length) < 1e-6
    assert session.objects == before and session.document_revision == revision
    # A changed start tangent must not silently rotate the linked section.
    rotated = PathDefinition(
        (ControlPointSpline(((0, 0, 0), (10, 0, 0), (20, 10, 0), (30, 10, 0))),)
    )
    invalid = session.update_path(path_id, rotated, expected_revision=revision)
    assert not invalid["ok"] and "plane normal" in invalid["error"]
    assert session.objects == before
    session.cad_document.adapter.profile_operations(session)._refresh()
    assert session.document_revision == revision
    wrong = SweepDefinition(CadProfile("wrong", profile_id), CadPath("wrong", path_id))
    assert not session.apply_modeling(wrong, expected_revision=revision)["ok"]


def loft_alignment_and_order(directory):
    session = CadSession(execution_mode="embedded")
    check(session.new_document("LoftAlignment"))
    import math

    points = (
        (10, 0, 0),
        (10, 10, 0),
        (0, 10, 0),
        (-10, 10, 0),
        (-10, 0, 0),
        (-10, -10, 0),
        (0, -10, 0),
        (10, -10, 0),
        (10, 0, 0),
    )
    weights = (
        1,
        math.sqrt(0.5),
        1,
        math.sqrt(0.5),
        1,
        math.sqrt(0.5),
        1,
        math.sqrt(0.5),
        1,
    )
    knots = (0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1)

    def profile_spec(poles, height):
        return ProfileDefinition(
            (
                ControlPointSpline(
                    poles, degree=2, closed=True, weights=weights, knots=knots
                ),
            ),
            frame=CoordinateFrame(origin=(0, 0, height)),
        )

    first = profile_spec(points, 0)
    second = profile_spec(tuple(reversed(points)), 10)
    a = check(
        session.apply_modeling(first, expected_revision=session.document_revision)
    )["object_id"]
    b = check(
        session.apply_modeling(second, expected_revision=session.document_revision)
    )["object_id"]
    count = len(session.cad_document.native_handle.Objects)
    rejected = session.loft(
        profile_ids=[a, b], expected_revision=session.document_revision
    )
    assert not rejected["ok"] and "winding" in rejected["error"]
    assert len(session.cad_document.native_handle.Objects) == count
    # Same traversal but shifted start point is also ambiguous within one edge.
    shifted = (*points[2:-1], *points[:3])
    shifted_profile = profile_spec(shifted, 10)
    c = check(
        session.apply_modeling(
            shifted_profile, expected_revision=session.document_revision
        )
    )["object_id"]
    rejected = session.loft(profile_ids=[a, c])
    assert not rejected["ok"] and "seam" in rejected["error"]
    aligned = profile_spec(points, 10)
    b = check(
        session.apply_modeling(aligned, expected_revision=session.document_revision)
    )["object_id"]
    result = check(
        session.loft(
            profile_ids=[a, b], ruled=True, expected_revision=session.document_revision
        )
    )
    native = session.runtime_objects[result["object_id"]].native_handle
    assert [obj.RapidCADObjectId for obj in native.Sections] == [a, b]
    # OCC integrates the rational loft surface numerically; allow 1% while
    # detecting the collapsed/twisted result from inconsistent correspondence.
    assert abs(native.Shape.Volume / (math.pi * 1000) - 1) < 0.01, native.Shape.Volume
    assert result["alignment"] == "automatic"
    # Section order remains exactly the caller's order.
    reverse = check(session.loft(profile_ids=[b, a], ruled=True))
    assert [
        obj.RapidCADObjectId
        for obj in session.runtime_objects[reverse["object_id"]].native_handle.Sections
    ] == [b, a]
    preserve = LoftDefinition(
        (session.profile_references[a], session.profile_references[b]),
        alignment="preserve",
    )
    assert (
        session.inspect_modeling_support(preserve)["support"]["status"] == "unsupported"
    )
    assert (
        session.apply_modeling(preserve, expected_revision=session.document_revision)[
            "error_code"
        ]
        == "cad_operation_not_supported"
    )
    assert not session.loft(profile_ids=[a, b], alignment="preserve")["ok"]


def typed_segments_and_feature_edits(directory):
    from rapidcadpy import (
        ArcSegment,
        CircleSegment,
        LineSegment,
        LoftFeatureUpdate,
    )

    session = CadSession(execution_mode="embedded")
    created = check(session.new_document("HarnessContracts"))
    first = check(
        session.create_profile(
            definition=ProfileDefinition((CircleSegment((0, 0, 0), 5),)),
            expected_revision=created["document_revision"],
        )
    )
    second = check(
        session.create_profile(
            definition=ProfileDefinition(
                (CircleSegment((0, 0, 0), 7),), frame=CoordinateFrame(origin=(0, 0, 10))
            ),
            expected_revision=first["document_revision"],
        )
    )
    loft = check(
        session.loft(
            profile_ids=[first["object_id"], second["object_id"]],
            expected_revision=second["document_revision"],
        )
    )
    identity = loft["object_id"]
    native = session.runtime_objects[identity].native_handle
    name = native.Name
    updated = check(
        session.update_feature(
            identity,
            LoftFeatureUpdate(
                make_solid=False,
                ruled=True,
                profile_ids=(second["object_id"], first["object_id"]),
            ),
            expected_revision=loft["document_revision"],
        )
    )
    assert updated["object_id"] == identity and updated["object_type"] == "shell"
    assert (
        updated["created_object_ids"] == []
        and identity in updated["changed_object_ids"]
    )
    assert native.Name == name and native.Ruled and not native.Solid
    assert [obj.RapidCADObjectId for obj in native.Sections] == [
        second["object_id"],
        first["object_id"],
    ]
    assert not session.update_feature(
        identity,
        {"kind": "loft", "Solid": True},
        expected_revision=updated["document_revision"],
    )["ok"]
    assert not session.update_feature(
        identity,
        {"kind": "sweep", "make_solid": True},
        expected_revision=updated["document_revision"],
    )["ok"]
    assert not native.Solid
    # Mixed typed line/arc segments build a connected native path.
    path = check(
        session.create_path(
            definition=PathDefinition(
                (
                    LineSegment((0, 0, 0), (0, 10, 0)),
                    ArcSegment((0, 10, 0), (3, 13, 0), (6, 10, 0)),
                )
            ),
            expected_revision=session.document_revision,
        )
    )
    assert session.runtime_objects[path["object_id"]].native_handle.GeometryCount == 2
    caps = check(session.inspect_capabilities())
    assert (
        "update_feature" in caps["operations"]
        and caps["restrictions"]["path_dimensions"] == "planar_local_xy"
    )
    inspection = check(session.inspect_geometry())
    assert inspection["geometry_valid"] and inspection["has_model_geometry"]
    report = next(obj for obj in inspection["objects"] if obj["object_id"] == identity)
    assert set(report["dependency_ids"]) == {first["object_id"], second["object_id"]}
    assert report["dimensions"]["z"] == 10 and report["dimensions"]["unit"] == "mm"
    assert report["operation_support"]["update_feature"]["status"] == "conditional"
    json.dumps(inspection, allow_nan=False)
    broken = session.cad_document.native_handle.addObject("Part::Loft", "BrokenLoft")
    session.cad_document.native_handle.recompute()
    invalid = check(session.inspect_geometry())
    assert not invalid["geometry_valid"]
    assert any(
        obj["errors"]
        for obj in invalid["objects"]
        if session.runtime_objects[obj["object_id"]].native_name == broken.Name
    )


def sweep_feature_edits_and_typed_profile_updates(directory):
    from rapidcadpy import CircleSegment, SweepFeatureUpdate

    session, profile_id, path_id, _definition = spline_sweep_setup()
    sweep = check(session.sweep(profile_id, path_id))
    identity = sweep["object_id"]
    native = session.runtime_objects[identity].native_handle
    updated = check(
        session.update_feature(
            identity,
            SweepFeatureUpdate(
                orientation="frenet", transition="round", make_solid=False
            ),
            expected_revision=sweep["document_revision"],
        )
    )
    assert native.Frenet and native.Transition == "Round corner" and not native.Solid
    assert updated["object_type"] == "shell" and updated["object_id"] == identity
    profile = ProfileDefinition(
        (CircleSegment((0, 0, 0), 2),), frame=CoordinateFrame(normal=(0, 1, 0))
    )
    resized = check(
        session.update_profile(
            profile_id,
            definition=profile.to_dict(),
            expected_revision=updated["document_revision"],
        )
    )
    assert {identity, profile_id} <= set(resized["changed_object_ids"])
    assert native.Sections[0].RapidCADObjectId == profile_id
    assert not session.update_feature(
        identity,
        SweepFeatureUpdate(make_solid=True),
        expected_revision=updated["document_revision"],
    )["ok"]
    assert not native.Solid
    before = copy.deepcopy(session.objects)
    original_validate = FreeCADMutationBackend.validate

    def broken_validate(self):
        raise RuntimeError("injected feature update failure")

    try:
        FreeCADMutationBackend.validate = broken_validate
        failure = session.update_feature(
            identity,
            SweepFeatureUpdate(make_solid=True, orientation="corrected_frenet"),
            expected_revision=resized["document_revision"],
        )
    finally:
        FreeCADMutationBackend.validate = original_validate
    assert not failure["ok"] and not native.Solid and native.Frenet
    assert session.objects == before
    report = check(session.inspect_geometry([identity]))["objects"][0]
    assert set(report["dependency_ids"]) == {path_id, profile_id}


def bridge_server_contract_guard(directory):
    from rapidcadpy.bridge_contract import bridge_contract
    from rapidcadpy.integrations.freecad import gui_bridge

    gui_bridge._refresh_gui = lambda **kwargs: None
    check(gui_bridge._dispatch("ping", {}))
    assert not gui_bridge._dispatch("create_profile", {}, None)["ok"]
    assert not gui_bridge._dispatch(
        "create_profile", {}, {**bridge_contract(["create_profile"]), "major": 2}
    )["ok"]
    assert gui_bridge._dispatch(
        "negotiate_contract", bridge_contract(["create_profile"])
    )["ok"]
    session = CadSession(execution_mode="embedded")
    check(session.new_document("BridgeGuard"))
    gui_bridge._SESSION = session
    from rapidcadpy import CircleSegment

    request = ProfileDefinition((CircleSegment((0, 0, 0), 2),)).to_dict()
    result = check(
        gui_bridge._dispatch(
            "create_profile",
            {"definition": request, "expected_revision": session.document_revision},
            bridge_contract(["create_profile"]),
        )
    )
    assert result["profile"]["object_id"] in session.profile_references
    # A cached older session class cannot advertise a new contract just because
    # the bridge module on disk was updated.
    session.LIVE_CONTRACT_VERSION = None
    assert gui_bridge._dispatch("ping", {})["bridge_contract"]["operations"] == []
    assert not gui_bridge._dispatch(
        "inspect_capabilities", {}, bridge_contract(["inspect_capabilities"])
    )["ok"]


if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2])
    print("native scenario passed")
