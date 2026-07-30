"""FreeCAD integration for RapidCADPy.

FreeCAD's Python modules are loaded lazily so discovery and connector
installation remain usable from a normal Python interpreter.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "FreeCADApp": (".app", "FreeCADApp"),
    "ensure_freecad_python_path": (".app", "ensure_freecad_python_path"),
    "FreeCADGuiConnection": (".gui_connection", "FreeCADGuiConnection"),
    "FreeCADShape": (".shape", "FreeCADShape"),
    "FreeCADSketch2D": (".sketch2d", "FreeCADSketch2D"),
    "FreeCADWorkplane": (".workplane", "FreeCADWorkplane"),
    "discover_freecad_user_mod_dir": (
        ".connector_addon",
        "discover_freecad_user_mod_dir",
    ),
    "install_freecad_connector": (
        ".connector_addon",
        "install_freecad_connector",
    ),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
