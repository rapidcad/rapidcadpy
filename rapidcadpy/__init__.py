"""
rapidcadpy - A Python library for fluent CAD API.

This package provides a fluent API for CAD modeling operations.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core fluent API components
try:
    from .app import App
except ImportError:
    App = None

# Core geometry types for fluent API
from .cad_types import Vector, Vertex

# Core shape and sketch classes
from .shape import Shape
from .sketch2d import Sketch2D
from .sketch3d import Sketch3D

# Components - preset profiles
from .components import profiles

# Optional integrations are loaded lazily so importing rapidcadpy does not
# require FreeCAD/OCP/Inventor runtimes or emit warnings in MCP stdio mode.
_OPTIONAL_INTEGRATIONS = {
    "OpenCascadeApp": ".integrations.occ.app",
    "OpenCascadeOcpApp": ".integrations.ocp.app",
    "InventorApp": ".integrations.inventor.app",
    "FreeCADApp": ".integrations.freecad.app",
}


def __getattr__(name):
    if name not in _OPTIONAL_INTEGRATIONS:
        raise AttributeError(f"module 'rapidcadpy' has no attribute {name!r}")

    import importlib

    module = importlib.import_module(_OPTIONAL_INTEGRATIONS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


# Essential primitives for fluent modeling
from .workplane import Workplane

# Handle optional dependencies that might not be available
try:
    # Import any modules that depend on optional dependencies like torch
    pass
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings

    warnings.warn(f"Some optional rapidcadpy modules could not be imported: {e}")

# Define what gets imported with "from rapidcadpy import *"
__all__ = [
    # Core fluent API
    "App",
    "Workplane",
    "Shape",
    "Sketch2D",
    "Sketch3D",
    # Components
    "profiles",
    # Optional integrations
    "OpenCascadeApp",
    "OpenCascadeOcpApp",
    "InventorApp",
    "FreeCADApp",
]
