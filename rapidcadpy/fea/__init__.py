"""
FEA module for rapidcadpy - Finite Element Analysis integration

This module requires additional dependencies:
    pip install rapidcadpy[fea]
"""

# Always export base classes (no torch dependency needed)
from .kernels.base import FEAKernel
from .load_case.load_case import LoadCase
from .fea_analyzer import FEAAnalyzer

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

    try:
        from .kernels.torch_fem_kernel import TorchFEMKernel
    except Exception as _torch_fem_err:
        # TorchFEM import may fail due to CuPy/CUDA issues
        # Make TorchFEMKernel available but warn if used
        import warnings

        warnings.warn(f"TorchFEMKernel not available due to: {_torch_fem_err}")
        TorchFEMKernel = None

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
