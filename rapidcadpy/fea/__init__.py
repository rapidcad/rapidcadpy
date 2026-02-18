"""
FEA module for rapidcadpy - Finite Element Analysis integration

This module requires additional dependencies:
    pip install rapidcadpy[fea]
"""

# Always export base classes (no torch dependency needed)
from .kernels.base import FEAKernel, FEAAnalyzer
from .load_case import LoadCase

try:
    import torch
    import pyvista

    # Export public API
    from .materials import Material, MaterialProperties, CustomMaterial
    from .boundary_conditions import (
        FixedConstraint,
        CylindricalConstraint,
        PinnedConstraint,
        RollerConstraint,
        DistributedLoad,
        PointLoad,
        PressureLoad,
        visualize_boundary_conditions,
    )
    from .results import FEAResults, OptimizationResult
    from .kernels.torch_fem_kernel import TorchFEMKernel

    __all__ = [
        "FEAKernel",
        "FEAAnalyzer",
        "LoadCase",
        "TorchFEMKernel",
        "Material",
        "MaterialProperties",
        "CustomMaterial",
        "FixedConstraint",
        "CylindricalConstraint",
        "PinnedConstraint",
        "RollerConstraint",
        "DistributedLoad",
        "PointLoad",
        "PressureLoad",
        "visualize_boundary_conditions",
        "FEAResults",
        "OptimizationResult",
    ]

    _FEA_AVAILABLE = True

except ImportError as e:
    import warnings

    warnings.warn(
        f"FEA module dependencies not available. "
        f"Install with: pip install rapidcadpy[fea]\n"
        f"Missing: {e}"
    )

    # Provide stub implementations that raise helpful errors
    class _FEANotAvailable:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"FEA functionality requires additional dependencies")

    Material = _FEANotAvailable
    MaterialProperties = _FEANotAvailable
    CustomMaterial = _FEANotAvailable
    FixedConstraint = _FEANotAvailable
    PinnedConstraint = _FEANotAvailable
    RollerConstraint = _FEANotAvailable
    DistributedLoad = _FEANotAvailable
    PointLoad = _FEANotAvailable
    PressureLoad = _FEANotAvailable
    FEAResults = _FEANotAvailable
    OptimizationResult = _FEANotAvailable
    TorchFEMKernel = _FEANotAvailable

    __all__ = ["FEAKernel", "FEAAnalyzer", "LoadCase"]
    _FEA_AVAILABLE = False
