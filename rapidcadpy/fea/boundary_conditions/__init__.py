"""
Boundary conditions for FEA analysis.

This package provides loads and constraints that can be applied to FEA models.

Structure
---------
boundary_conditions/
├── constraints.py          BoundaryCondition (ABC), FixedConstraint,
│                           CylindricalConstraint, PinnedConstraint, RollerConstraint
├── _visualization.py       visualize_boundary_conditions
└── loads/
    ├── base.py             Load (ABC)
    ├── distributed.py      DistributedLoad
    ├── point.py            PointLoad
    ├── pressure.py         PressureLoad
    ├── concentrated.py     ConcentratedLoad
    ├── linear_distributed.py  LinearDistributedLoad
    └── acceleration.py     AccelerationLoad

All names are re-exported here so existing code that does::

    from rapidcadpy.fea.boundary_conditions import FixedConstraint, DistributedLoad

continues to work without modification.
"""

# Constraints
from .constraints import (
    BoundaryCondition,
    FixedConstraint,
    CylindricalConstraint,
    PinnedConstraint,
    RollerConstraint,
)

# Loads
from .loads import (
    Load,
    DistributedLoad,
    PointLoad,
    PressureLoad,
    ConcentratedLoad,
    LinearDistributedLoad,
    AccelerationLoad,
)

# Visualization
from ._visualization import visualize_boundary_conditions

__all__ = [
    # constraints
    "BoundaryCondition",
    "FixedConstraint",
    "CylindricalConstraint",
    "PinnedConstraint",
    "RollerConstraint",
    # loads
    "Load",
    "DistributedLoad",
    "PointLoad",
    "PressureLoad",
    "ConcentratedLoad",
    "LinearDistributedLoad",
    "AccelerationLoad",
    # visualization
    "visualize_boundary_conditions",
]
