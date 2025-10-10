"""
rapidcadpy - A Python library for fluent CAD API.

This package provides a fluent API for CAD modeling operations.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core fluent API components
from .app import App, OpenCascadeApp
from .cad import Cad

# Core geometry types for fluent API
from .cad_types import Vector, Vertex

# Essential primitives for fluent modeling
from .primitive import Arc, Circle, Line
from .workplane import Workplane

# Optional integrations - import with error handling
try:
    from .integrations.inventor.app import InventorApp
except ImportError:
    InventorApp = None

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
    "Cad",
    # Geometry types
    "Vector",
    "Vertex",
    # Primitives
    "Line",
    "Arc",
    "Circle",
    # Optional integrations
    "OpenCascadeApp",
    "InventorApp",
]
