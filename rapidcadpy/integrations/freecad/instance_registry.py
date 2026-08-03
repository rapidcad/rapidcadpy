"""Per-user discovery records for RapidCADPy bridges inside FreeCAD."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class FreeCADInstance:
    """Connection information for one bridge-enabled FreeCAD GUI."""

    instance_id: str
    pid: int
    host: str
    port: int
    token: str
    info_path: Path
    started_at: float
    source: str = "freecad_addon"
    package_root: str = ""

    def public_dict(self) -> Dict[str, Any]:
        """Return discovery metadata without the bridge authentication token."""
        return {
            "instance_id": self.instance_id,
            "pid": self.pid,
            "host": self.host,
            "port": self.port,
            "started_at": self.started_at,
            "source": self.source,
            "package_root": self.package_root,
        }


def instance_registry_dir() -> Path:
    """Return the private directory used to advertise attachable instances."""
    configured = os.environ.get("RAPIDCADPY_FREECAD_INSTANCE_DIR", "").strip()
    if configured:
        directory = Path(configured).expanduser().resolve()
    else:
        home = Path.home()
        if sys.platform == "darwin":
            directory = (
                Path("/private/tmp") / f"rapidcadpy-{os.getuid()}" / "freecad-instances"
            ).resolve()
        elif os.name == "nt":
            local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
            base = (
                Path(local_app_data).expanduser()
                if local_app_data
                else home / "AppData" / "Local"
            )
            directory = (base / "RapidCADPy" / "FreeCAD" / "instances").resolve()
        else:
            xdg_state_home = os.environ.get("XDG_STATE_HOME", "").strip()
            base = (
                Path(xdg_state_home).expanduser()
                if xdg_state_home
                else home / ".local" / "state"
            )
            directory = (base / "rapidcadpy" / "freecad" / "instances").resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        directory.chmod(0o700)
    except OSError:
        pass
    return directory


def instance_info_path(instance_id: str) -> Path:
    """Return the registry path for an already-sanitized instance identifier."""
    safe_id = _safe_instance_id(instance_id)
    return instance_registry_dir() / f"{safe_id}.json"


def instance_ipc_dir(instance_id: str) -> Path:
    """Return the private filesystem IPC directory for one FreeCAD instance."""
    safe_id = _safe_instance_id(instance_id)
    directory = instance_registry_dir() / f"{safe_id}.ipc"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        directory.chmod(0o700)
    except OSError:
        pass
    return directory


def _safe_instance_id(instance_id: str) -> str:
    safe_id = "".join(
        character
        for character in str(instance_id)
        if character.isalnum() or character in {"-", "_"}
    )
    if not safe_id:
        raise ValueError("instance_id must contain at least one safe character.")
    return safe_id


def write_instance_record(
    *,
    instance_id: str,
    pid: int,
    host: str,
    port: int,
    token: str,
    source: str,
    package_root: str,
    info_path: Optional[Path] = None,
) -> Path:
    """Atomically write a permission-restricted bridge discovery record."""
    path = (info_path or instance_info_path(instance_id)).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = {
        "instance_id": instance_id,
        "pid": int(pid),
        "host": str(host),
        "port": int(port),
        "token": str(token),
        "started_at": time.time(),
        "source": str(source),
        "package_root": str(package_root),
    }
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        temporary_path.chmod(0o600)
    except OSError:
        pass
    temporary_path.replace(path)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def read_instance(path: Path) -> FreeCADInstance:
    """Parse and validate one bridge discovery record."""
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    host = str(payload["host"])
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError(f"Refusing non-loopback FreeCAD bridge host: {host}")
    port = int(payload["port"])
    if not 0 < port < 65536:
        raise ValueError(f"Invalid FreeCAD bridge port: {port}")
    token = str(payload["token"])
    if not token:
        raise ValueError("FreeCAD bridge token is empty.")
    pid = int(payload["pid"])
    return FreeCADInstance(
        instance_id=str(payload.get("instance_id") or f"freecad-{pid}"),
        pid=pid,
        host=host,
        port=port,
        token=token,
        info_path=resolved,
        started_at=float(payload.get("started_at", resolved.stat().st_mtime)),
        source=str(payload.get("source", "freecad_addon")),
        package_root=str(payload.get("package_root", "")),
    )


def process_is_running(pid: int) -> bool:
    """Return whether a process appears to exist without signaling it."""
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def discover_instance_records(remove_stale: bool = True) -> List[FreeCADInstance]:
    """Return live-looking instance records ordered newest first."""
    instances: List[FreeCADInstance] = []
    for path in instance_registry_dir().glob("*.json"):
        try:
            instance = read_instance(path)
        except Exception:
            continue
        if process_is_running(instance.pid):
            instances.append(instance)
            continue
        if remove_stale:
            try:
                path.unlink()
            except OSError:
                pass
    return sorted(instances, key=lambda item: item.started_at, reverse=True)


def remove_instance_record(path: Path) -> None:
    """Remove one discovery record if it still exists."""
    try:
        path.expanduser().resolve().unlink()
    except FileNotFoundError:
        pass
