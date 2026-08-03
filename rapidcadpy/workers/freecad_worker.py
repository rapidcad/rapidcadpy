"""Persistent FreeCAD worker for a standalone RapidCADPy CAD session."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

freecad_lib_path = os.environ.get("FREECAD_LIB_PATH", "").strip()
if freecad_lib_path and freecad_lib_path not in sys.path:
    sys.path.insert(0, freecad_lib_path)

# Reserve original stdout for JSON-lines protocol, then route all normal
# FreeCAD/Python stdout/stderr noise to a log file. Some CAD exports print from
# Python/C++ and would otherwise corrupt the JSON-lines protocol.
WORKER_LOG_PATH = os.environ.get(
    "RAPIDCADPY_CAD_WORKER_LOG", "/tmp/rapidcadpy_freecad_worker.log"
)
Path(WORKER_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
_protocol_fd = os.dup(1)
_worker_log = open(WORKER_LOG_PATH, "a", encoding="utf-8", buffering=1)
os.dup2(_worker_log.fileno(), 1)
os.dup2(_worker_log.fileno(), 2)
sys.stdout = os.fdopen(_protocol_fd, "w", buffering=1)
sys.stderr = _worker_log

from rapidcadpy.cad_session import CadSession  # noqa: E402


SESSION = CadSession()


def dispatch(method: str, params: dict) -> dict:
    if method == "shutdown":
        return {"ok": True, "summary": "Worker shutting down", "_shutdown": True}
    fn = getattr(SESSION, method, None)
    if fn is None:
        return {"ok": False, "error": f"Unknown worker method '{method}'"}
    try:
        return fn(**params)
    except Exception:
        return {"ok": False, "error": traceback.format_exc()}


def main() -> int:
    _worker_log.write("=== RapidCADPy FreeCAD worker start ===\n")
    _worker_log.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = dispatch(
                request.get("method", ""), request.get("params", {}) or {}
            )
        except Exception:
            response = {"ok": False, "error": traceback.format_exc()}
        shutdown = response.pop("_shutdown", False)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
        if shutdown:
            _worker_log.write("=== RapidCADPy FreeCAD worker shutdown ===\n")
            _worker_log.flush()
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
