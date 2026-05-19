"""
FreeCAD integration for rapidcadpy.

Requires FreeCAD Python modules to be importable. If FreeCAD is not installed
as a standard Python package, point FREECAD_LIB_PATH to the directory that
contains FreeCAD.so / FreeCAD.pyd (e.g. /usr/lib/freecad/lib or the Mod
directory from a conda-forge freecad install).
"""

from .app import FreeCADApp, ensure_freecad_python_path

ensure_freecad_python_path()

try:
    import FreeCAD  # noqa: F401
    import Part  # noqa: F401
except ImportError as _e:
    import logging as _logging

    _logging.warning(
        f"FreeCAD integration: could not import FreeCAD/Part ({_e}). "
        "Set FREECAD_LIB_PATH to FreeCAD module directory and use a Python "
        "runtime ABI-compatible with FreeCAD build."
    )

from .shape import FreeCADShape
from .sketch2d import FreeCADSketch2D
from .workplane import FreeCADWorkplane

__all__ = ["FreeCADApp", "FreeCADShape", "FreeCADSketch2D", "FreeCADWorkplane"]
