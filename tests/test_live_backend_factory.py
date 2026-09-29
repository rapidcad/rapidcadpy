"""A second fake CAD integration exercises the live abstract-factory boundary."""

import pytest
from rapidcadpy.cad_session import CadSession
from rapidcadpy.integrations.freecad.live_backend import FreeCADLiveBackendFactory
from rapidcadpy.live_backend import (
    CadApplication,
    LiveCadBackendFactory,
    LiveCadBackendRegistry,
)
from rapidcadpy.modeling import (
    CadPath,
    CadProfile,
    InterpolatedSpline,
    LoftDefinition,
    PathDefinition,
    ProfileDefinition,
    SweepDefinition,
)
from rapidcadpy.operation_result import CadOperationResult, CadOperationSupport


def profile():
    return ProfileDefinition(
        (InterpolatedSpline(((0, 0, 0), (1, 2, 0)), degree=1),), closed=False
    )


class FakeConnection:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.reachable = True

    def call(self, method, params):
        self.calls.append((method, params))
        return {"ok": self.reachable, "active_document": {"name": "Fake"}}

    def close(self):
        self.closed = True


class FakeServices:
    def __init__(self, connection):
        self.connection = connection
        self.applied = []

    def inspect_support(self, definition):
        return CadOperationSupport(
            "create_profile", "supported", "native_feature", "Fake support"
        )

    def hydrate_active_document(self):
        self.connection.call("hydrate", {})
        return {"ok": True, "document_revision": "r1", "document": {"name": "Fake"}}

    def apply(self, definition, *, expected_revision):
        self.applied.append((definition, expected_revision))
        if expected_revision != "r1":
            return CadOperationResult.failure(
                "Stale revision", error_code="document_revision_mismatch"
            )
        return CadOperationResult.success(
            document_revision="r2",
            created_object_ids=("profile1",),
            support=self.inspect_support(definition),
            data={"profile": {"document_id": "doc", "object_id": "profile1"}},
        )


class FakeFactory(LiveCadBackendFactory):
    def __init__(self):
        self.connection = FakeConnection()
        self.services = FakeServices(self.connection)
        self.application = CadApplication("fake", "Fake CAD", "fake-1")
        self.requested_target = None

    def discover(self):
        return (
            self.application,
            CadApplication("fake", "Fake CAD", "offline", connected=False),
        )

    def connect(self, target_id=None):
        self.requested_target = target_id
        return self.application, self.connection

    def create_modeling(self, connection):
        assert connection is self.connection
        return self.services

    def create_hydrator(self, connection):
        assert connection is self.connection
        return self.services

    def create_capability_inspector(self, connection):
        assert connection is self.connection
        return self.services


def session_with_fake():
    registry = LiveCadBackendRegistry(include_defaults=False)
    factory = FakeFactory()
    registry.register("fake", lambda: factory)
    return (
        CadSession(execution_mode="gui", live_backend_registry=registry),
        factory,
        registry,
    )


def test_selected_factory_routes_discovery_hydration_modeling_and_legacy_calls():
    session, factory, _ = session_with_fake()
    discovered = session.list_cad_applications()
    assert discovered["supported_software"] == ["fake"]
    assert discovered["application_count"] == discovered["unreachable_count"] == 1
    selected = session.select_cad_application(" FAKE ", "fake-1")
    assert selected["document_revision"] == "r1"
    assert factory.requested_target == "fake-1"
    assert session.get_active_cad_application()["software"] == "fake"
    assert session.use_active_document()["software"] == "fake"
    request = profile()
    assert session.inspect_modeling_support(request)["support"]["status"] == "supported"
    result = session.apply_modeling(request, expected_revision="r1")
    assert result["created_object_ids"] == ["profile1"]
    assert session.document_revision == "r2"
    assert factory.services.applied == [(request, "r1")]
    session.rect(10, 20)
    assert factory.connection.calls[-1][0] == "rect"
    assert session.live_backend.modeling is factory.services
    session.select_object("profile1")
    assert factory.connection.calls[-1] == ("select_object", {"object_id": "profile1"})
    session.fit_view()
    assert factory.connection.calls[-1] == ("fit_view", {})


def test_failed_selection_preserves_previous_backend_and_closes_candidate():
    session, original, registry = session_with_fake()
    session.select_cad_application("fake")
    broken = FakeFactory()
    broken.connection.reachable = False
    registry.register("broken", lambda: broken)
    assert session.select_cad_application("broken")["ok"] is False
    assert session.live_backend.connection is original.connection
    assert broken.connection.closed
    assert not original.connection.closed
    assert not session.select_cad_application("missing")["ok"]
    assert session.live_backend.connection is original.connection


def test_modeling_requires_selection_and_revision():
    session, factory, _ = session_with_fake()
    assert not session.apply_modeling(profile(), expected_revision="r1")["ok"]
    session.select_cad_application("fake")
    assert not session.apply_modeling(profile(), expected_revision="")["ok"]
    assert factory.services.applied == []
    result = session.apply_modeling(profile(), expected_revision="stale")
    assert result["error_code"] == "document_revision_mismatch"
    assert session.document_revision == "r1"


def test_factory_product_failure_closes_connection():
    class BrokenFactory(FakeFactory):
        def create_hydrator(self, connection):
            raise RuntimeError("Cannot construct hydrator")

    factory = BrokenFactory()
    with pytest.raises(RuntimeError):
        factory.create()
    assert factory.connection.closed


def test_freecad_new_contract_is_explicitly_unsupported_without_native_execution():
    connection = FakeConnection()
    factory = FreeCADLiveBackendFactory()
    support = factory.create_capability_inspector(connection).inspect_support(profile())
    result = factory.create_modeling(connection).apply(
        profile(), expected_revision="r1"
    )
    assert support.status == "unsupported"
    assert result.support == support
    assert result.error_code == "cad_operation_not_supported"
    assert connection.calls == []


def test_registry_rejects_duplicates_and_is_instance_scoped():
    first = LiveCadBackendRegistry(include_defaults=False)
    first.register(" FAKE ", FakeFactory)
    with pytest.raises(ValueError, match="already"):
        first.register("fake", FakeFactory)
    with pytest.raises(ValueError, match="Unsupported"):
        LiveCadBackendRegistry().factory("fake")


@pytest.mark.parametrize(
    "definition",
    [
        LoftDefinition(
            (CadProfile("doc", "p1"), CadProfile("doc", "p2")), make_solid=False
        ),
        SweepDefinition(CadProfile("doc", "p1"), CadPath("doc", "path")),
        PathDefinition(
            (InterpolatedSpline(((0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0))),)
        ),
        ProfileDefinition(
            (InterpolatedSpline(((0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0))),),
            closed=False,
        ),
    ],
)
def test_freecad_modeling_dispatches_serialized_intent_and_result(definition):
    support = CadOperationSupport(
        "loft", "supported", "native_dependent_feature", "Native loft"
    )
    expected = CadOperationResult.success(
        document_revision="r2",
        created_object_ids=("loft1",),
        support=support,
        data={"object_id": "loft1", "object_type": "shell"},
    )

    class LoftConnection(FakeConnection):
        def call(self, method, params):
            self.calls.append((method, params))
            if method == "inspect_modeling_support":
                return {"ok": True, "support": support.to_dict()}
            return expected.to_dict()

    connection = LoftConnection()
    factory = FreeCADLiveBackendFactory()
    assert (
        factory.create_capability_inspector(connection).inspect_support(definition)
        == support
    )
    assert (
        factory.create_modeling(connection).apply(definition, expected_revision="r1")
        == expected
    )
    assert connection.calls == [
        ("inspect_modeling_support", {"definition": definition.to_dict()}),
        (
            "apply_modeling",
            {"definition": definition.to_dict(), "expected_revision": "r1"},
        ),
    ]


@pytest.mark.parametrize(
    "definition",
    [
        LoftDefinition(
            (CadProfile("doc", "p1"), CadProfile("doc", "p2")), alignment="preserve"
        ),
        PathDefinition(
            (InterpolatedSpline(((0, 0, 0), (1, 1, 0), (2, 2, 0)), degree=2),)
        ),
        PathDefinition(
            (InterpolatedSpline(((0, 0, 0), (1, 1, 1), (2, 2, 0), (3, 3, 0))),)
        ),
    ],
)
def test_freecad_unsupported_modes_do_not_contact_the_application(definition):
    connection = FakeConnection()
    factory = FreeCADLiveBackendFactory()
    support = factory.create_capability_inspector(connection).inspect_support(
        definition
    )
    assert support.status == "unsupported"
    result = factory.create_modeling(connection).apply(
        definition, expected_revision="r1"
    )
    assert result.error_code == "cad_operation_not_supported"
    assert result.support == support
    assert connection.calls == []
