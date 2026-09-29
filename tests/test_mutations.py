"""Contract tests for atomic mutation order and runtime/native rollback."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from rapidcadpy.cad_session import CadSession, SemanticObject
from rapidcadpy.mutations import SessionMutationState, document_revision, run_mutation
from rapidcadpy.operation_result import CadOperationSupport


@pytest.mark.parametrize(
    "failure",
    [
        "input",
        "begin",
        "mutate",
        "recompute",
        "validate",
        "hydrate",
        "serialize",
        "commit",
        None,
    ],
)
def test_mutation_is_atomic_through_response_serialization(failure):
    session = CadSession(execution_mode="embedded")
    workplane = SimpleNamespace(_pending_shapes=["original"], _accumulated_loops=[])
    session.objects["wp"] = SemanticObject("wp", "workplane", "WP", "original")
    session.runtime_objects["wp"] = workplane
    native = {}
    events = []
    refresh_count = 0

    def refresh():
        nonlocal refresh_count
        refresh_count += 1
        events.append("refresh")
        if failure == "hydrate" and refresh_count == 2:
            session.objects.clear()
            raise RuntimeError("hydration failed midway")
        session.objects = {"wp": SemanticObject("wp", "workplane", "WP", "original")}
        for key in native:
            session.objects[key] = SemanticObject(key, "profile", key, "hydrated")
        session.document_revision = document_revision(
            session.objects, session.parameters
        )

    refresh()
    original = deepcopy(session.objects)
    revision = session.document_revision
    events.clear()
    refresh_count = 0

    class Backend:
        def begin(self, name):
            events.append("begin")
            if failure == "begin":
                raise RuntimeError("begin failed")
            self.saved = deepcopy(native)

        def recompute(self):
            events.append("recompute")
            if failure == "recompute":
                raise RuntimeError("recompute failed")

        def validate(self):
            events.append("validate")
            if failure == "validate":
                raise RuntimeError("invalid shape")

        def commit(self):
            events.append("commit")
            if failure == "commit":
                raise RuntimeError("commit failed")

        def abort(self):
            events.append("abort")
            native.clear()
            native.update(self.saved)

    def inputs():
        events.append("input")
        if failure == "input":
            raise ValueError("invalid inputs")

    def mutate():
        events.append("mutate")
        native["profile"] = "new native sketch"
        workplane._pending_shapes.clear()
        session._profile_cache["new"] = "profile"
        if failure == "mutate":
            raise RuntimeError("creation failed")
        return {
            "object_id": "profile",
            **({"leak": object()} if failure == "serialize" else {}),
        }

    result = run_mutation(
        state=SessionMutationState(session, refresh),
        backend=Backend(),
        support=CadOperationSupport(
            "create_profile", "supported", "native_feature", "test"
        ),
        expected_revision=revision,
        validate_inputs=inputs,
        mutate=mutate,
    )
    if failure:
        assert not result.ok
        assert native == {}
        assert session.objects == original
        assert session.document_revision == revision
        assert session.runtime_objects["wp"] is workplane
        assert workplane._pending_shapes == ["original"]
        assert session._profile_cache == {}
        assert (
            result.created_object_ids
            == result.changed_object_ids
            == result.removed_object_ids
            == ()
        )
    else:
        assert result.ok
        assert result.created_object_ids == ("profile",)
        assert result.document_revision != revision
        assert events == [
            "refresh",
            "input",
            "begin",
            "mutate",
            "recompute",
            "validate",
            "refresh",
            "commit",
        ]


def test_revision_ignores_history_and_workplanes_but_detects_native_changes():
    session = CadSession()
    session.objects["profile"] = SemanticObject(
        "profile", "profile", "P", "first", metadata={"geometry": {"area": 10}}
    )
    first = document_revision(session.objects, {})
    session.objects["profile"].source_op = "rehydrated"
    session.objects["wp"] = SemanticObject("wp", "workplane", "WP", "transient")
    assert document_revision(session.objects, {}) == first
    session.objects["profile"].metadata["geometry"]["area"] = 20
    assert document_revision(session.objects, {}) != first
