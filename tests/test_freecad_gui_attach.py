"""Tests for discovering and attaching to an existing FreeCAD GUI."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapidcadpy.integrations.freecad.connector_addon import (
    install_freecad_connector,
)
from rapidcadpy.integrations.freecad.app import FreeCADApp
from rapidcadpy.integrations.freecad.gui_connection import FreeCADGuiConnection
from rapidcadpy.integrations.freecad.instance_registry import (
    instance_ipc_dir,
    write_instance_record,
)
from rapidcadpy.integrations.freecad import gui_connection as gui_connection_module
from rapidcadpy.cad_session import CadSession


def test_connector_installer_creates_idempotent_freecad_module(tmp_path):
    mod_dir = tmp_path / "Mod"
    package_root = Path(__file__).resolve().parents[1]

    first = install_freecad_connector(
        mod_dir=str(mod_dir),
        package_root=str(package_root),
    )
    second = install_freecad_connector(
        mod_dir=str(mod_dir),
        package_root=str(package_root),
    )

    init_gui = mod_dir / "RapidCADConnector" / "InitGui.py"
    namespace_init_gui = (
        mod_dir / "RapidCADConnector" / "freecad" / "rapidcad_connector" / "init_gui.py"
    )
    assert first["ok"] is True
    assert second["ok"] is True
    assert first["restart_required"] is True
    assert "start_persistent_bridge" in init_gui.read_text(encoding="utf-8")
    assert "start_persistent_bridge" in namespace_init_gui.read_text(encoding="utf-8")
    assert str(mod_dir / "RapidCADConnector" / "_vendor") in init_gui.read_text(
        encoding="utf-8"
    )
    assert (
        mod_dir / "RapidCADConnector" / "_vendor" / "rapidcadpy" / "cad_session.py"
    ).is_file()


def test_connector_installer_does_not_overwrite_unmanaged_addon(tmp_path):
    addon_dir = tmp_path / "Mod" / "RapidCADConnector"
    addon_dir.mkdir(parents=True)
    (addon_dir / "InitGui.py").write_text("# user code\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="unmanaged"):
        install_freecad_connector(
            mod_dir=str(tmp_path / "Mod"),
            package_root=str(Path(__file__).resolve().parents[1]),
        )


def test_discover_attach_and_hydrate_running_freecad(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPIDCADPY_FREECAD_INSTANCE_DIR", str(tmp_path))
    write_instance_record(
        instance_id=f"freecad-{os.getpid()}",
        pid=os.getpid(),
        host="127.0.0.1",
        port=54321,
        token="test-token",
        source="test",
        package_root=str(Path(__file__).resolve().parents[1]),
    )

    def fake_call(connection, method, params):
        assert connection.token == "test-token"
        if method == "ping":
            return {
                "ok": True,
                "pid": os.getpid(),
                "freecad_version": "1.0.2",
                "active_document": {
                    "name": "Bracket",
                    "label": "Bracket",
                    "file_name": "/tmp/bracket.FCStd",
                    "object_count": 3,
                },
            }
        if method == "use_active_document":
            return {
                "ok": True,
                "summary": "Hydrated active FreeCAD document with 3 objects",
                "document": {"name": "Bracket"},
                "document_revision": "sha256:test",
                "object_count": 3,
                "objects": [],
                "tree": [],
            }
        return {"ok": False, "error": "Unknown method"}

    monkeypatch.setattr(FreeCADGuiConnection, "call", fake_call)

    instances = FreeCADGuiConnection.list_instances()
    app = FreeCADApp("attach")
    session = CadSession(execution_mode="gui")
    attached = session.attach_freecad()

    assert len(instances) == 1
    assert instances[0]["active_document"]["name"] == "Bracket"
    assert app.is_remote is True
    assert app.call("ping")["ok"] is True
    assert attached["ok"] is True
    assert attached["execution_mode"] == "freecad_gui_attach"
    assert attached["instance_id"] == f"freecad-{os.getpid()}"
    assert attached["document"]["name"] == "Bracket"
    assert attached["document_revision"] == "sha256:test"


def test_generic_application_discovery_and_selection_use_freecad_adapter(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("RAPIDCADPY_FREECAD_INSTANCE_DIR", str(tmp_path))
    instance_id = f"freecad-{os.getpid()}"
    write_instance_record(
        instance_id=instance_id,
        pid=os.getpid(),
        host="127.0.0.1",
        port=54321,
        token="test-token",
        source="test",
        package_root=str(Path(__file__).resolve().parents[1]),
    )

    def fake_call(connection, method, params):
        if method == "ping":
            return {
                "ok": True,
                "pid": os.getpid(),
                "active_document": {"name": "Bracket", "object_count": 1},
            }
        if method == "use_active_document":
            return {
                "ok": True,
                "summary": "Hydrated active document",
                "document": {"name": "Bracket"},
                "document_revision": "sha256:generic",
                "object_count": 1,
                "objects": [],
                "tree": [],
            }
        return {"ok": False, "error": f"Unexpected method: {method}"}

    monkeypatch.setattr(FreeCADGuiConnection, "call", fake_call)
    session = CadSession(execution_mode="gui")

    listed = session.list_cad_applications()
    selected = session.select_cad_application(
        software="freecad",
        target_id=instance_id,
    )
    active = session.get_active_cad_application()

    assert listed["application_count"] == 1
    assert listed["applications"][0]["software"] == "freecad"
    assert listed["applications"][0]["target_id"] == instance_id
    assert "feature.extrude" in listed["applications"][0]["capabilities"]
    assert selected["software"] == "freecad"
    assert selected["target_id"] == instance_id
    assert selected["document_revision"] == "sha256:generic"
    assert active["selected"] is True
    assert active["target_id"] == instance_id


def test_discovery_reports_advertised_but_unreachable_freecad(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("RAPIDCADPY_FREECAD_INSTANCE_DIR", str(tmp_path))
    write_instance_record(
        instance_id=f"freecad-{os.getpid()}",
        pid=os.getpid(),
        host="127.0.0.1",
        port=54321,
        token="test-token",
        source="test",
        package_root=str(Path(__file__).resolve().parents[1]),
    )
    monkeypatch.setattr(
        FreeCADGuiConnection,
        "call",
        lambda connection, method, params: {
            "ok": False,
            "error": "PermissionError: Operation not permitted",
        },
    )

    result = CadSession(execution_mode="gui").list_freecad_instances()

    assert result["ok"] is True
    assert result["instance_count"] == 0
    assert result["discovered_count"] == 1
    assert result["unreachable_count"] == 1
    assert result["unreachable_instances"][0]["connected"] is False
    assert "PermissionError" in result["unreachable_instances"][0]["connection_error"]


def test_connection_falls_back_to_private_filesystem_ipc(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("RAPIDCADPY_FREECAD_INSTANCE_DIR", str(tmp_path))
    instance_id = f"freecad-{os.getpid()}"
    connection = FreeCADGuiConnection(
        process=None,
        host="127.0.0.1",
        port=54321,
        token="test-token",
        pid=os.getpid(),
        instance_id=instance_id,
    )
    ipc_dir = instance_ipc_dir(instance_id)

    def reject_localhost(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted")

    def respond_via_files():
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            request_paths = list(ipc_dir.glob("*.request.json"))
            if request_paths:
                request_path = request_paths[0]
                request = json.loads(request_path.read_text(encoding="utf-8"))
                assert request["token"] == "test-token"
                assert request["method"] == "ping"
                request_id = request_path.name.removesuffix(".request.json")
                response_path = ipc_dir / f"{request_id}.response.json"
                response_path.write_text(
                    json.dumps({"ok": True, "pid": os.getpid()}),
                    encoding="utf-8",
                )
                return
            time.sleep(0.005)
        raise AssertionError("Filesystem IPC request was not written.")

    monkeypatch.setattr(
        gui_connection_module.socket,
        "create_connection",
        reject_localhost,
    )
    responder = threading.Thread(target=respond_via_files)
    responder.start()

    result = connection.call("ping", {})

    responder.join(timeout=2)
    assert not responder.is_alive()
    assert result["ok"] is True
    assert result["pid"] == os.getpid()
    assert result["bridge_transport"] == "filesystem"


def test_gui_modeling_auto_attaches_and_creates_a_document(monkeypatch):
    class FakeWorker:
        def __init__(self):
            self.calls = []

        def call(self, method, params):
            self.calls.append((method, params))
            return {"ok": True, "summary": f"Completed {method}"}

    worker = FakeWorker()
    session = CadSession(execution_mode="gui")

    def attach_freecad(instance_id=None, use_active_document=True):
        assert instance_id is None
        assert use_active_document is True
        session._worker = worker
        session._gui_connection = worker
        return {
            "ok": True,
            "summary": "Attached without an active document",
            "active_document": None,
            "warning": "FreeCAD has no active document.",
        }

    monkeypatch.setattr(session, "attach_freecad", attach_freecad)

    result = session.work_plane("XY")

    assert result["ok"] is True
    assert worker.calls == [
        ("new_document", {"name": "RapidCADPy"}),
        ("work_plane", {"plane": "XY", "offset": None}),
    ]


def test_live_code_execution_auto_attaches_and_delegates(monkeypatch):
    class FakeWorker:
        def __init__(self):
            self.calls = []

        def call(self, method, params):
            self.calls.append((method, params))
            return {
                "ok": True,
                "summary": "Executed code visibly in the attached FreeCAD GUI",
            }

    worker = FakeWorker()
    session = CadSession(execution_mode="gui")

    def attach_freecad(instance_id=None, use_active_document=True):
        assert instance_id is None
        assert use_active_document is False
        session._worker = worker
        session._gui_connection = worker
        return {"ok": True, "summary": "Attached"}

    monkeypatch.setattr(session, "attach_freecad", attach_freecad)

    result = session.execute_code("import FreeCAD")

    assert result["ok"] is True
    assert worker.calls == [("execute_code", {"code": "import FreeCAD"})]


def test_embedded_session_adopts_active_document_without_creating_one(monkeypatch):
    document = SimpleNamespace(
        Name="Unsaved",
        Label="Unsaved model",
        FileName="",
        Objects=[],
        recompute=lambda: None,
    )
    monkeypatch.setitem(
        sys.modules, "FreeCAD", SimpleNamespace(ActiveDocument=document)
    )
    session = CadSession(execution_mode="embedded")

    result = session.use_active_document()

    assert result["ok"] is True
    assert result["document"]["name"] == "Unsaved"
    assert result["document"]["file_name"] == ""
    assert result["object_count"] == 0
    assert session.app.get_doc() is document
