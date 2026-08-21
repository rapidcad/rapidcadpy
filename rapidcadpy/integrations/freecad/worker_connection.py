"""Out-of-process FreeCAD worker transport for :class:`CadSession`.

Runs a persistent FreeCAD Python interpreter as a subprocess and speaks a
JSON-lines protocol to it. Used as a fallback when FreeCAD cannot be imported
directly into the host process.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from typing import Any, Dict


class FreeCADWorkerClient:
    """JSON-lines client for a persistent FreeCAD Python worker."""

    def __init__(self) -> None:
        package_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        worker_path = os.path.join(
            package_root,
            "rapidcadpy",
            "workers",
            "freecad_worker.py",
        )
        freecad_python = os.environ.get(
            "FREECAD_PYTHON",
            "/Applications/FreeCAD.app/Contents/Resources/bin/python",
        )
        env = os.environ.copy()
        env["RAPIDCADPY_CAD_WORKER"] = "1"
        env.setdefault(
            "RAPIDCADPY_CAD_WORKER_LOG", "/tmp/rapidcadpy_freecad_worker.log"
        )
        env.setdefault(
            "FREECAD_LIB_PATH", "/Applications/FreeCAD.app/Contents/Resources/lib"
        )
        env["PYTHONPATH"] = (
            package_root
            if not env.get("PYTHONPATH")
            else package_root + os.pathsep + env["PYTHONPATH"]
        )
        self.proc = subprocess.Popen(
            [freecad_python, worker_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        self.timeout_seconds = float(
            os.environ.get("RAPIDCADPY_CAD_WORKER_TIMEOUT", "120")
        )

    def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.proc.poll() is not None:
            return self._dead_response()
        if self.proc.stdin is None or self.proc.stdout is None:
            return {"ok": False, "error": "FreeCAD worker pipes unavailable."}

        request = {"method": method, "params": params}
        try:
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
            ready, _, _ = select.select(
                [self.proc.stdout], [], [], self.timeout_seconds
            )
            if not ready:
                return {
                    "ok": False,
                    "error": (
                        f"FreeCAD worker timed out after {self.timeout_seconds:g}s "
                        f"while running '{method}'. Check "
                        f"{os.environ.get('RAPIDCADPY_CAD_WORKER_LOG', '/tmp/rapidcadpy_freecad_worker.log')}."
                    ),
                }
            line = self.proc.stdout.readline()
        except Exception as exc:
            return {"ok": False, "error": f"FreeCAD worker call failed: {exc}"}

        if not line:
            return self._dead_response()
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            return {
                "ok": False,
                "error": f"FreeCAD worker returned non-JSON output: {exc}: {line[:500]}",
            }

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.call("shutdown", {})
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _dead_response(self) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": (
                f"FreeCAD worker exited with code {self.proc.returncode}. Check "
                f"{os.environ.get('RAPIDCADPY_CAD_WORKER_LOG', '/tmp/rapidcadpy_freecad_worker.log')}."
            ),
        }


def ensure_package_import_path() -> None:
    """Add the RapidCADPy source root when running a bundled worker directly."""
    package_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
