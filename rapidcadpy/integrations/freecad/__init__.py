"""
FreeCAD integration for rapidcadpy.

Requires FreeCAD Python modules to be importable. If FreeCAD is not installed
as a standard Python package, point FREECAD_LIB_PATH to the directory that
contains FreeCAD.so / FreeCAD.pyd (e.g. /usr/lib/freecad/lib or the Mod
directory from a conda-forge freecad install).
"""

import os
import sys

# Attempt to load FreeCAD modules from the optional env-var path
_freecad_lib_path = os.environ.get("FREECAD_LIB_PATH")
if _freecad_lib_path and _freecad_lib_path not in sys.path:
    sys.path.insert(0, _freecad_lib_path)

try:
    import FreeCAD  # noqa: F401
    import Part  # noqa: F401
except ImportError as _e:
    import logging as _logging

    _logging.warning(
        f"FreeCAD integration: could not import FreeCAD/Part ({_e}). "
        "Set the FREECAD_LIB_PATH environment variable to the directory "
        "containing FreeCAD.so / FreeCAD.pyd."
    )

from .app import FreeCADApp
from .shape import FreeCADShape
from .sketch2d import FreeCADSketch2D
from .workplane import FreeCADWorkplane

__all__ = ["FreeCADApp", "FreeCADShape", "FreeCADSketch2D", "FreeCADWorkplane"]
