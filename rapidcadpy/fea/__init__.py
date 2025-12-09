"""
FEA module for rapidcadpy - Finite Element Analysis integration

This module requires additional dependencies:
    pip install rapidcadpy[fea]
"""

# Always export base classes (no torch dependency needed)
from rapidcadpy.fea.kernels.base import FEAKernel, FEAAnalyzer

try:
    import torch
    import pyvista

    # Export public API
    from rapidcadpy.fea.materials import Material, MaterialProperties, CustomMaterial
    from rapidcadpy.fea.boundary_conditions import (
        FixedConstraint,
        PinnedConstraint,
        RollerConstraint,
        DistributedLoad,
        PointLoad,
        PressureLoad,
        visualize_boundary_conditions,
    )
    from rapidcadpy.fea.results import FEAResults, OptimizationResult
    from rapidcadpy.fea.kernels.torch_fem_kernel import TorchFEMKernel

    __all__ = [
        "FEAKernel",
        "FEAAnalyzer",
        "TorchFEMKernel",
        "Material",
        "MaterialProperties",
        "CustomMaterial",
        "FixedConstraint",
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
            raise ImportError(
                "FEA functionality requires additional dependencies. "
                "Install with: pip install rapidcadpy[fea]"
            )

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

    __all__ = ["FEAKernel", "FEAAnalyzer"]
    _FEA_AVAILABLE = False
