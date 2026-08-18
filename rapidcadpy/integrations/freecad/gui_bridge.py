"""Authenticated RapidCADPy command bridge running inside the FreeCAD GUI."""

from __future__ import annotations

import json
import logging
import os
import queue
import secrets
import socketserver
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import FreeCAD as App
import FreeCADGui as Gui

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from rapidcadpy.integrations.freecad.instance_registry import (  # noqa: E402
    instance_ipc_dir,
    remove_instance_record,
    write_instance_record,
)
from rapidcadpy.cad_session import CadSession  # noqa: E402

try:
    from PySide import QtCore
except ImportError:
    from PySide2 import QtCore


_REQUESTS: "queue.Queue[Dict[str, Any]]" = queue.Queue()
_SESSION = CadSession(execution_mode="embedded")
_TOKEN = ""
_SERVER: Optional["_BridgeServer"] = None
_SERVER_THREAD: Optional[threading.Thread] = None
_TIMER: Any = None
_INFO_PATHS: list[Path] = []
_IPC_DIR: Optional[Path] = None
_INSTANCE_ID = f"freecad-{os.getpid()}"
_START_LOCK = threading.Lock()
_LOG_PATH = Path(
    os.environ.get("RAPIDCADPY_GUI_BRIDGE_LOG", "/tmp/rapidcadpy_freecad_gui_bridge.log")
)
_LOGGER = logging.getLogger("rapidcadpy.freecad.gui_bridge")


def _configure_logging() -> None:
    """Persist minimal bridge lifecycle diagnostics outside transient chat UI."""

    if _LOGGER.handlers:
        return
    try:
        handler = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    except OSError:
        return
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


def _log_request(event: str, method: str, **fields: Any) -> None:
    _configure_logging()
    try:
        _LOGGER.info(
            json.dumps({"event": event, "method": method, **fields}, default=str)
        )
    except Exception:
        pass


def _active_document_info() -> Optional[Dict[str, Any]]:
    document = App.ActiveDocument
    if document is None:
        return None
    return {
        "name": str(getattr(document, "Name", "")),
        "label": str(getattr(document, "Label", "")),
        "file_name": str(getattr(document, "FileName", "")),
        "object_count": len(getattr(document, "Objects", [])),
    }


def _native_object(object_id: str):
    runtime_object = _SESSION.runtime_objects.get(object_id)
    if runtime_object is None:
        raise KeyError(f"Unknown object_id '{object_id}'")
    native = getattr(runtime_object, "native_handle", None)
    if native is None and getattr(runtime_object, "feature", None) is not None:
        native = runtime_object.feature.native_handle
    if native is None:
        raise TypeError(f"Object '{object_id}' has no native FreeCAD handle.")
    return native


def _refresh_gui(fit: bool = False) -> None:
    document = App.ActiveDocument
    if document is not None:
        document.recompute()
    Gui.updateGui()
    active_gui_document = Gui.activeDocument()
    if fit and active_gui_document is not None:
        active_gui_document.activeView().fitAll()
        Gui.updateGui()


def _dispatch(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if method == "ping":
        version = getattr(App, "Version", lambda: [])()
        return {
            "ok": True,
            "summary": "FreeCAD GUI bridge ready",
            "pid": os.getpid(),
            "instance_id": _INSTANCE_ID,
            "gui": True,
            "freecad_version": ".".join(str(part) for part in version[:3]),
            "active_document": _active_document_info(),
        }
    if method == "select_object":
        object_id = str(params["object_id"])
        native = _native_object(object_id)
        Gui.Selection.clearSelection()
        Gui.Selection.addSelection(native.Document.Name, native.Name)
        _refresh_gui()
        return {
            "ok": True,
            "summary": f"Selected {object_id} in FreeCAD",
            "object_id": object_id,
        }
    if method == "fit_view":
        _refresh_gui(fit=True)
        return {"ok": True, "summary": "Fit FreeCAD view to model"}

    function = getattr(_SESSION, method, None)
    if function is None:
        return {"ok": False, "error": f"Unknown GUI bridge method '{method}'"}
    result = function(**params)
    if result.get("ok"):
        fit = method in {
            "open_document",
            "use_active_document",
            "new_document",
            "execute_code",
            "extrude",
            "hole",
            "cut",
            "fillet",
            "generate_drawing",
        }
        _refresh_gui(fit=fit)
    return result


def _drain_requests() -> None:
    for _ in range(25):
        try:
            request = _REQUESTS.get_nowait()
        except queue.Empty:
            break
        method = str(request["method"])
        _log_request("dispatch_started", method)
        try:
            request["response"] = _dispatch(
                method,
                dict(request.get("params") or {}),
            )
        except Exception:
            request["response"] = {
                "ok": False,
                "error": traceback.format_exc(),
            }
        finally:
            _log_request(
                "dispatch_finished",
                method,
                ok=bool(request.get("response", {}).get("ok")),
            )
            request["event"].set()
    _drain_file_requests()


def _drain_file_requests() -> None:
    """Process sandbox-compatible request files on FreeCAD's GUI thread."""
    if _IPC_DIR is None:
        return
    for request_path in sorted(_IPC_DIR.glob("*.request.json"))[:25]:
        request_id = request_path.name.removesuffix(".request.json")
        response_path = _IPC_DIR / f"{request_id}.response.json"
        temporary_path = _IPC_DIR / f".{request_id}.response.tmp"
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            if request.get("token") != _TOKEN:
                response = {"ok": False, "error": "Invalid bridge token."}
            else:
                response = _dispatch(
                    str(request.get("method", "")),
                    dict(request.get("params") or {}),
                )
        except Exception:
            response = {"ok": False, "error": traceback.format_exc()}
        try:
            temporary_path.write_text(json.dumps(response), encoding="utf-8")
            try:
                temporary_path.chmod(0o600)
            except OSError:
                pass
            temporary_path.replace(response_path)
        finally:
            try:
                request_path.unlink()
            except FileNotFoundError:
                pass


class _BridgeHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline()
        try:
            request = json.loads(line.decode("utf-8"))
            if request.get("token") != _TOKEN:
                response = {"ok": False, "error": "Invalid bridge token."}
            else:
                _log_request("request_received", str(request.get("method", "")))
                pending: Dict[str, Any] = {
                    "method": request.get("method", ""),
                    "params": request.get("params", {}),
                    "event": threading.Event(),
                }
                _REQUESTS.put(pending)
                timeout = float(os.environ.get("RAPIDCADPY_GUI_COMMAND_TIMEOUT", "120"))
                if not pending["event"].wait(timeout=timeout):
                    response = {
                        "ok": False,
                        "error": "Timed out waiting for FreeCAD's GUI thread.",
                    }
                else:
                    response = pending["response"]
        except Exception:
            response = {"ok": False, "error": traceback.format_exc()}
        _log_request(
            "response_sent",
            str(request.get("method", "")) if "request" in locals() else "",
            ok=bool(response.get("ok")),
        )
        self.wfile.write(json.dumps(response).encode("utf-8") + b"\n")


class _BridgeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_bridge(
    *,
    token: Optional[str] = None,
    handshake_path: Optional[Path] = None,
    source: str = "freecad_addon",
) -> Dict[str, Any]:
    """Start or return the bridge and advertise it for external attachment."""
    global _SERVER, _SERVER_THREAD, _TIMER, _TOKEN, _INFO_PATHS, _IPC_DIR

    with _START_LOCK:
        if _SERVER is not None:
            return {
                "ok": True,
                "instance_id": _INSTANCE_ID,
                "pid": os.getpid(),
                "host": _SERVER.server_address[0],
                "port": _SERVER.server_address[1],
            }

        _TOKEN = token or secrets.token_urlsafe(32)
        ipc_dir = instance_ipc_dir(_INSTANCE_ID)
        for stale_path in ipc_dir.glob("*.json"):
            try:
                stale_path.unlink()
            except OSError:
                pass
        server = _BridgeServer(("127.0.0.1", 0), _BridgeHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        registry_path = write_instance_record(
            instance_id=_INSTANCE_ID,
            pid=os.getpid(),
            host=str(server.server_address[0]),
            port=int(server.server_address[1]),
            token=_TOKEN,
            source=source,
            package_root=str(_PACKAGE_ROOT),
        )
        info_paths = [registry_path]
        if handshake_path is not None:
            resolved_handshake = handshake_path.expanduser().resolve()
            if resolved_handshake != registry_path:
                write_instance_record(
                    instance_id=_INSTANCE_ID,
                    pid=os.getpid(),
                    host=str(server.server_address[0]),
                    port=int(server.server_address[1]),
                    token=_TOKEN,
                    source=source,
                    package_root=str(_PACKAGE_ROOT),
                    info_path=resolved_handshake,
                )
                info_paths.append(resolved_handshake)

        timer = QtCore.QTimer()
        timer.timeout.connect(_drain_requests)
        timer.start(20)

        application = QtCore.QCoreApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(stop_bridge)

        _SERVER = server
        _SERVER_THREAD = server_thread
        _TIMER = timer
        _INFO_PATHS = info_paths
        _IPC_DIR = ipc_dir

    return {
        "ok": True,
        "instance_id": _INSTANCE_ID,
        "pid": os.getpid(),
        "host": server.server_address[0],
        "port": server.server_address[1],
    }


def start_persistent_bridge() -> Dict[str, Any]:
    """Start the attachable bridge from the installed FreeCAD addon."""
    return start_bridge(source="freecad_addon")


def stop_bridge() -> None:
    """Stop the bridge and remove only this process's discovery records."""
    global _SERVER, _SERVER_THREAD, _TIMER, _INFO_PATHS, _IPC_DIR
    server = _SERVER
    _SERVER = None
    if _TIMER is not None:
        _TIMER.stop()
        _TIMER = None
    if server is not None:
        server.shutdown()
        server.server_close()
    _SERVER_THREAD = None
    for path in _INFO_PATHS:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("pid", -1)) == os.getpid():
                remove_instance_record(path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            pass
    _INFO_PATHS = []
    _IPC_DIR = None


_ENV_INFO_PATH = os.environ.get("RAPIDCADPY_GUI_BRIDGE_INFO", "").strip()
_ENV_TOKEN = os.environ.get("RAPIDCADPY_GUI_BRIDGE_TOKEN", "").strip()
if _ENV_INFO_PATH and _ENV_TOKEN:
    start_bridge(
        token=_ENV_TOKEN,
        handshake_path=Path(_ENV_INFO_PATH),
        source="external_launch",
    )
