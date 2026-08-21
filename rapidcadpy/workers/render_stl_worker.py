"""Isolated STL-to-PNG renderer used by the FreeCAD worker fallback.

VTK/pyvista rendering can crash when initialized from a background thread on
macOS. Running it in this short-lived process keeps rendering on the main
thread and prevents a native renderer failure from terminating its caller.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")


def main() -> int:
    if len(sys.argv) != 6:
        print(
            "usage: render_stl_worker.py INPUT.stl OUTPUT.png VIEW WIDTH HEIGHT",
            file=sys.stderr,
        )
        return 2

    input_path, output_path, view, width, height = sys.argv[1:]
    try:
        from rapidcadpy.integrations.freecad.shape import _render_stl

        _render_stl(
            input_path,
            output_path,
            view,
            int(width),
            int(height),
        )
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
