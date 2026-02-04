"""
rapidcadpy - A Python library for fluent CAD API.

This package provides a fluent API for CAD modeling operations.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core fluent API components
from .app import App

# Core geometry types for fluent API
from .cad_types import Vector, Vertex

# Core shape and sketch classes
from .shape import Shape
from .sketch2d import Sketch2D
from .sketch3d import Sketch3D

# Components - preset profiles
from .components import profiles

# Optional integrations - import with error handling
try:
    from .integrations.occ.app import OpenCascadeApp
except ImportError:
    OpenCascadeApp = None

try:
    from .integrations.ocp.app import OpenCascadeOcpApp
except ImportError:
    OpenCascadeOcpApp = None

try:
    from .integrations.inventor.app import InventorApp
except ImportError as e:
    print(f"Warning: Could not import InventorApp due to: {e}")
    InventorApp = None

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
]
