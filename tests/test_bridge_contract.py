"""Contract negotiation and client-side exclusion of outdated connectors."""

import json

import pytest
from rapidcadpy.bridge_contract import (
    advertised_capabilities,
    bridge_contract,
    negotiate_contract,
)
from rapidcadpy.integrations.freecad.capabilities import FREECAD_CAPABILITIES
from rapidcadpy.integrations.freecad.gui_connection import FreeCADGuiConnection


@pytest.mark.parametrize(
    "server",
    [
        {},
        None,
        {"name": "other", "major": 1, "minor": 0, "operations": ["sweep"]},
        {"name": "rapidcad.live", "major": 2, "minor": 0, "operations": ["sweep"]},
        bridge_contract(["loft"]),
    ],
)
def test_old_or_incomplete_contract_does_not_advertise_sweeps(server):
    capabilities = advertised_capabilities(FREECAD_CAPABILITIES, server)
    assert "feature.sweep" not in capabilities
    assert "feature.extrude" in capabilities
    assert not negotiate_contract(bridge_contract(["sweep"]), server or {})["ok"]


def test_compatible_contract_advertises_only_available_operations():
    server = bridge_contract(["sweep", "inspect_capabilities"])
    assert negotiate_contract(bridge_contract(["sweep"]), server)["ok"]
    capabilities = advertised_capabilities(FREECAD_CAPABILITIES, server)
    assert "feature.sweep" in capabilities and "capability.inspect" in capabilities
    assert "profile.create" not in capabilities and "feature.loft" not in capabilities


@pytest.mark.parametrize("advertisement", [None, bridge_contract(["sweep"])])
def test_connection_gates_and_transmits_contract(monkeypatch, advertisement):
    calls = []

    class Stream:
        def write(self, data):
            self.request = json.loads(data)
            calls.append(self.request)

        def flush(self):
            pass

        def readline(self):
            response = {"ok": True, "pid": 1}
            if self.request["method"] == "ping" and advertisement is not None:
                response["bridge_contract"] = advertisement
            return json.dumps(response).encode() + b"\n"

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, timeout):
            pass

        def makefile(self, mode):
            return Stream()

    monkeypatch.setattr("socket.create_connection", lambda *args, **kwargs: Socket())
    connection = FreeCADGuiConnection(
        None, "127.0.0.1", 1234, "dummy-test-token", pid=1
    )
    result = connection.call(
        "sweep", {"profile_id": "p", "path_id": "path", "expected_revision": "r1"}
    )
    if advertisement is None:
        assert result["error_code"] == "cad_bridge_incompatible"
        assert [call["method"] for call in calls] == ["ping"]
        assert "feature.sweep" not in connection.capabilities
    else:
        assert result["ok"]
        assert [call["method"] for call in calls] == ["ping", "sweep"]
        assert calls[-1]["contract"] == bridge_contract(["sweep"])
        assert "feature.sweep" in connection.capabilities


def test_server_minor_version_must_meet_client_requirement():
    client = {**bridge_contract(["sweep"]), "minor": 1}
    assert not negotiate_contract(client, bridge_contract(["sweep"]))["ok"]
    assert negotiate_contract(client, {**bridge_contract(["sweep"]), "minor": 1})["ok"]


@pytest.mark.parametrize("advertisement", [None, bridge_contract(["sweep"])])
def test_filesystem_fallback_uses_the_same_contract(
    monkeypatch, tmp_path, advertisement
):
    import threading
    import time

    from rapidcadpy.integrations.freecad import gui_connection

    calls = []
    stop = threading.Event()
    monkeypatch.setattr(gui_connection, "instance_ipc_dir", lambda _: tmp_path)

    def blocked_socket(*args, **kwargs):
        raise OSError("injected TCP unavailable")

    monkeypatch.setattr("socket.create_connection", blocked_socket)

    def respond():
        while not stop.is_set():
            for path in tmp_path.glob("*.request.json"):
                request = json.loads(path.read_text())
                response_path = path.with_name(
                    path.name.replace(".request.json", ".response.json")
                )
                if response_path.exists():
                    continue
                calls.append(request)
                response = {"ok": True, "pid": 1}
                if request["method"] == "ping" and advertisement is not None:
                    response["bridge_contract"] = advertisement
                temporary = response_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(response))
                temporary.replace(response_path)
            time.sleep(0.005)

    thread = threading.Thread(target=respond, daemon=True)
    thread.start()
    try:
        connection = FreeCADGuiConnection(
            None, "127.0.0.1", 1234, "dummy-test-token", pid=1
        )
        connection.timeout_seconds = 1
        result = connection.call(
            "sweep", {"profile_id": "p", "path_id": "path", "expected_revision": "r1"}
        )
    finally:
        stop.set()
        thread.join(timeout=1)
    if advertisement is None:
        assert result["error_code"] == "cad_bridge_incompatible"
        assert [request["method"] for request in calls] == ["ping"]
    else:
        assert result["ok"] and result["bridge_transport"] == "filesystem"
        assert [request["method"] for request in calls] == ["ping", "sweep"]
        assert calls[-1]["contract"] == bridge_contract(["sweep"])
    assert not list(tmp_path.glob("*.json"))
