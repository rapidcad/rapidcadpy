"""Launch and communicate with a RapidCADPy bridge inside the FreeCAD GUI."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
from uuid import uuid4
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .instance_registry import (
    FreeCADInstance,
    discover_instance_records,
    instance_ipc_dir,
)


@dataclass(frozen=True)
class FreeCADInstallation:
    """Paths belonging to a discovered FreeCAD installation."""

    executable: Path
    python_executable: Optional[Path]
    lib_path: Optional[Path]
    app_bundle: Optional[Path] = None


def discover_freecad_installation() -> Optional[FreeCADInstallation]:
    """Discover the GUI executable and related Python paths."""
    configured_executable = os.environ.get("FREECAD_EXECUTABLE", "").strip()
    executable_candidates = [
        configured_executable,
        "/Applications/FreeCAD.app/Contents/MacOS/FreeCAD",
        shutil.which("FreeCAD") or "",
        shutil.which("freecad") or "",
    ]
    executable = next(
        (
            Path(candidate).expanduser().resolve()
            for candidate in executable_candidates
            if candidate and Path(candidate).expanduser().is_file()
        ),
        None,
    )
    if executable is None:
        return None

    configured_python = os.environ.get("FREECAD_PYTHON", "").strip()
    python_candidates = [
        configured_python,
        "/Applications/FreeCAD.app/Contents/Resources/bin/python",
    ]
    python_executable = next(
        (
            Path(candidate).expanduser().resolve()
            for candidate in python_candidates
            if candidate and Path(candidate).expanduser().is_file()
        ),
        None,
    )

    configured_lib = os.environ.get("FREECAD_LIB_PATH", "").strip()
    lib_candidates = [
        configured_lib,
        "/Applications/FreeCAD.app/Contents/Resources/lib",
        "/usr/lib/freecad/lib",
        "/usr/lib64/freecad/lib",
    ]
    lib_path = next(
        (
            Path(candidate).expanduser().resolve()
            for candidate in lib_candidates
            if candidate and Path(candidate).expanduser().is_dir()
        ),
        None,
    )
    app_bundle = (
        Path("/Applications/FreeCAD.app")
        if str(executable).startswith("/Applications/FreeCAD.app/")
        else None
    )
    return FreeCADInstallation(
        executable=executable,
        python_executable=python_executable,
        lib_path=lib_path,
        app_bundle=app_bundle,
    )


class FreeCADGuiConnection:
    """JSON-lines client for one bridge-enabled FreeCAD GUI process."""

    def __init__(
        self,
        process: Optional[subprocess.Popen],
        host: str,
        port: int,
        token: str,
        session_dir: Optional[Path] = None,
        *,
        pid: Optional[int] = None,
        instance_id: Optional[str] = None,
        info_path: Optional[Path] = None,
    ) -> None:
        self.process = process
        self.host = host
        self.port = int(port)
        self.token = token
        self.session_dir = session_dir
        self._pid = int(pid if pid is not None else process.pid if process else 0)
        self.instance_id = instance_id or f"freecad-{self._pid}"
        self.info_path = info_path
        self.timeout_seconds = float(
            os.environ.get("RAPIDCADPY_GUI_COMMAND_TIMEOUT", "120")
        )

    @classmethod
    def from_instance(cls, instance: FreeCADInstance) -> "FreeCADGuiConnection":
        """Connect to an instance that was advertised by the FreeCAD addon."""
        return cls(
            process=None,
            host=instance.host,
            port=instance.port,
            token=instance.token,
            session_dir=instance.info_path.parent,
            pid=instance.pid,
            instance_id=instance.instance_id,
            info_path=instance.info_path,
        )

    @classmethod
    def list_instances(cls) -> List[Dict[str, Any]]:
        """List advertised FreeCAD GUIs, including unreachable instances."""
        results: List[Dict[str, Any]] = []
        for instance in discover_instance_records():
            connection = cls.from_instance(instance)
            ping = connection.call("ping", {})
            public = instance.public_dict()
            if not ping.get("ok"):
                results.append(
                    {
                        **public,
                        "connected": False,
                        "connection_error": ping.get(
                            "error",
                            "FreeCAD bridge did not answer its health check.",
                        ),
                    }
                )
                continue
            if int(ping.get("pid", -1)) != instance.pid:
                results.append(
                    {
                        **public,
                        "connected": False,
                        "connection_error": (
                            "FreeCAD bridge PID did not match its discovery record."
                        ),
                    }
                )
                continue
            results.append(
                {
                    **public,
                    "connected": True,
                    "bridge_transport": ping.get("bridge_transport", "tcp"),
                    "active_document": ping.get("active_document"),
                    "freecad_version": ping.get("freecad_version"),
                }
            )
        return results

    @classmethod
    def attach(cls, instance_id: Optional[str] = None) -> "FreeCADGuiConnection":
        """Attach to one responsive GUI instance.

        With no explicit ID, attachment is automatic only when exactly one
        bridge-enabled FreeCAD process is available.
        """
        candidates: List[FreeCADGuiConnection] = []
        for instance in discover_instance_records():
            if instance_id is not None and instance.instance_id != instance_id:
                continue
            connection = cls.from_instance(instance)
            ping = connection.call("ping", {})
            if ping.get("ok") and int(ping.get("pid", -1)) == instance.pid:
                candidates.append(connection)

        if not candidates:
            suffix = f" with instance_id '{instance_id}'" if instance_id else ""
            raise RuntimeError(f"No attachable FreeCAD GUI found{suffix}.")
        if len(candidates) > 1:
            identifiers = ", ".join(item.instance_id for item in candidates)
            raise RuntimeError(
                "Multiple attachable FreeCAD GUIs found. Pass instance_id; "
                f"available IDs: {identifiers}"
            )
        return candidates[0]

    @classmethod
    def launch(cls, timeout_seconds: Optional[float] = None) -> "FreeCADGuiConnection":
        """Launch FreeCAD with the RapidCADPy bridge and wait until ready."""
        installation = discover_freecad_installation()
        if installation is None:
            raise RuntimeError(
                "FreeCAD GUI executable not found. Set FREECAD_EXECUTABLE."
            )

        session_dir = Path(tempfile.mkdtemp(prefix="rapidcadpy-freecad-gui-")).resolve()
        info_path = session_dir / "bridge.json"
        log_path = session_dir / "freecad-gui.log"
        token = secrets.token_urlsafe(32)
        bridge_path = Path(__file__).with_name("gui_bridge.py").resolve()
        package_root = Path(__file__).resolve().parents[3]

        env = os.environ.copy()
        env["RAPIDCADPY_GUI_BRIDGE_INFO"] = str(info_path)
        env["RAPIDCADPY_GUI_BRIDGE_TOKEN"] = token
        env["RAPIDCADPY_CAD_WORKER"] = "1"
        env["PYTHONPATH"] = (
            str(package_root)
            if not env.get("PYTHONPATH")
            else str(package_root) + os.pathsep + env["PYTHONPATH"]
        )
        if installation.lib_path is not None:
            env.setdefault("FREECAD_LIB_PATH", str(installation.lib_path))

        log_handle = log_path.open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                [str(installation.executable), str(bridge_path)],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
        finally:
            log_handle.close()

        wait_seconds = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else float(os.environ.get("RAPIDCADPY_GUI_LAUNCH_TIMEOUT", "45"))
        )
        deadline = time.monotonic() + wait_seconds
        last_error = ""
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"FreeCAD GUI exited with code {process.returncode}. "
                    f"See {log_path}."
                )
            if info_path.is_file():
                try:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                    connection = cls(
                        process=process,
                        host=str(info["host"]),
                        port=int(info["port"]),
                        token=token,
                        session_dir=session_dir,
                        pid=int(info.get("pid", process.pid)),
                        instance_id=str(
                            info.get("instance_id", f"freecad-{process.pid}")
                        ),
                        info_path=info_path,
                    )
                    ping = connection.call("ping", {})
                    if ping.get("ok"):
                        return connection
                    last_error = str(ping.get("error", "bridge ping failed"))
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.1)

        try:
            process.terminate()
        except Exception:
            pass
        raise RuntimeError(
            f"Timed out waiting {wait_seconds:g}s for the FreeCAD GUI bridge"
            + (f": {last_error}" if last_error else "")
            + f". See {log_path}."
        )

    def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        request = {
            "token": self.token,
            "method": method,
            "params": params,
        }
        socket_error: Optional[Exception] = None
        try:
            with socket.create_connection(
                (self.host, self.port), timeout=self.timeout_seconds
            ) as connection:
                connection.settimeout(self.timeout_seconds)
                stream = connection.makefile("rwb")
                stream.write(json.dumps(request).encode("utf-8") + b"\n")
                stream.flush()
                line = stream.readline()
        except Exception as exc:
            socket_error = exc
            file_response = self._call_via_files(method, params)
            if file_response.get("ok"):
                file_response.setdefault("bridge_transport", "filesystem")
                return file_response
            return {
                "ok": False,
                "error": (
                    f"FreeCAD GUI bridge call '{method}' failed over localhost "
                    f"TCP ({type(socket_error).__name__}: {socket_error}) and "
                    f"filesystem IPC ({file_response.get('error', 'unknown error')})."
                ),
            }
        if not line:
            return {
                "ok": False,
                "error": f"FreeCAD GUI bridge returned no response for '{method}'.",
            }
        try:
            response = json.loads(line.decode("utf-8"))
            response.setdefault("bridge_transport", "tcp")
            return response
        except Exception as exc:
            return {
                "ok": False,
                "error": f"Invalid FreeCAD GUI bridge response: {exc}",
            }

    def _call_via_files(
        self,
        method: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use private request/response files when localhost sockets are blocked."""
        request_id = uuid4().hex
        ipc_dir = instance_ipc_dir(self.instance_id)
        request_path = ipc_dir / f"{request_id}.request.json"
        response_path = ipc_dir / f"{request_id}.response.json"
        temporary_path = ipc_dir / f".{request_id}.{os.getpid()}.tmp"
        request = {
            "token": self.token,
            "method": method,
            "params": params,
        }
        timeout_seconds = (
            min(self.timeout_seconds, 2.0) if method == "ping" else self.timeout_seconds
        )
        deadline = time.monotonic() + timeout_seconds
        try:
            temporary_path.write_text(json.dumps(request), encoding="utf-8")
            try:
                temporary_path.chmod(0o600)
            except OSError:
                pass
            temporary_path.replace(request_path)

            while time.monotonic() < deadline:
                if response_path.is_file():
                    try:
                        return json.loads(response_path.read_text(encoding="utf-8"))
                    except Exception as exc:
                        return {
                            "ok": False,
                            "error": f"Invalid filesystem IPC response: {exc}",
                        }
                time.sleep(0.02)
            return {
                "ok": False,
                "error": (
                    f"timed out after {timeout_seconds:g}s waiting for the "
                    "FreeCAD filesystem bridge"
                ),
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            for path in (temporary_path, request_path, response_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def close(self) -> None:
        """Disconnect while deliberately leaving the launched GUI open."""

    @property
    def pid(self) -> int:
        return self._pid
