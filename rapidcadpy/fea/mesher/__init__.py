"""
Mesher module for FEA mesh generation.

Provides abstract base class and concrete implementations for different
meshing backends (Netgen, GMSH, etc.).
"""

from .base import MesherBase
from .netgen_mesher import NetgenMesher, import_geometry_netgen

__all__ = [
    'MesherBase',
    'NetgenMesher',
    'import_geometry_netgen',  # Backward compatibility
]
