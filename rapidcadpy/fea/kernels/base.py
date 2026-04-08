"""
FEA base classes with dependency injection architecture.

This module provides:
- FEAKernel: Abstract base class for FEA solver backends
- FEAAnalyzer: Non-abstract analyzer that uses a FEAKernel via dependency injection
"""

from abc import ABC, abstractmethod
import os
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
import pyvista as pv

from ..boundary_conditions import FixedConstraint, PointLoad  #
import logging

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from ...shape import Shape
    from ..load_case.load_case import LoadCase
    from ..materials import MaterialProperties
    from ..boundary_conditions import Load, BoundaryCondition
    from ..results import FEAResults, OptimizationResult


class FEAKernel(ABC):
    """
    Abstract base class for FEA solver backends.

    Concrete implementations (e.g., TorchFEMKernel) provide specific
    FEA solver backends. The kernel handles meshing, solving, and
    result extraction for a specific solver.
    """

    def __str__(self) -> str:
        """String representation of the FEA kernel."""
        solver_name = self.get_solver_name()
        available = "✓ available" if self.is_available() else "✗ not available"
        return f"FEAKernel({solver_name}, {available})"

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """
        Check if this kernel's dependencies are available.

        Returns:
            True if the kernel can be used, False otherwise
        """
        pass

    @classmethod
    @abstractmethod
    def get_solver_name(cls) -> str:
        """
        Get the name of the FEA solver backend.

        Returns:
            Name of the solver (e.g., "torch-fem", "CalculiX")
        """
        pass

    @abstractmethod
    def solve(
        self,
        shape: "Shape",
        material: "MaterialProperties",
        loads: List["Load"],
        constraints: List["BoundaryCondition"],
        mesh_size: float,
        element_type: str,
        verbose: bool = False,
    ) -> "FEAResults":
        """
        Run FEA analysis and return results.

        Args:
            shape: Shape to analyze
            material: Material properties
            loads: List of loads to apply
            constraints: List of boundary conditions to apply
            mesh_size: Target mesh element size (mm)
            element_type: Element type (solver-dependent)
            verbose: Print detailed progress information

        Returns:
            FEAResults object containing analysis results
        """
        pass

    def optimize(
        self,
        shape: "Shape",
        material: "MaterialProperties",
        loads: List["Load"],
        constraints: List["BoundaryCondition"],
        mesh_size: float,
        element_type: str,
        volume_fraction: float = 0.5,
        num_iterations: int = 100,
        penalization: float = 3.0,
        filter_radius: float = 0.0,
        move_limit: float = 0.2,
        rho_min: float = 1e-3,
        use_autograd: bool = False,
        verbose: bool = False,
    ) -> "OptimizationResult":
        """
        Run topology optimization using SIMP method.

        This is an optional method that kernels can implement for topology optimization.
        Not all kernels support optimization.

        Args:
            shape: Shape to optimize
            material: Material properties
            loads: List of loads to apply
            constraints: List of boundary conditions to apply
            mesh_size: Target mesh element size (mm)
            element_type: Element type (solver-dependent)
            volume_fraction: Target volume fraction (0 < v < 1)
            num_iterations: Number of optimization iterations
            penalization: SIMP penalization factor (p)
            filter_radius: Sensitivity filter radius (0 = no filtering)
            move_limit: Maximum change in density per iteration
            rho_min: Minimum density to avoid singularity
            use_autograd: Use automatic differentiation for sensitivities
            verbose: Print detailed progress information

        Returns:
            OptimizationResult object containing optimization results

        Raises:
            NotImplementedError: If the kernel does not support optimization
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support topology optimization"
        )
        pass

    def get_visualization_data(
        self,
        shape: Union["Shape", str],
        material: "MaterialProperties",
        loads: List["Load"],
        constraints: List["BoundaryCondition"],
        mesh_size: float,
        element_type: str,
        **kwargs,
    ) -> Tuple[pv.UnstructuredGrid, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mesh and boundary condition data for visualization.

        This method prepares the mesh and applies boundary conditions without solving,
        returning data needed for visualization of constraints and loads.

        Args:
            shape: Shape to visualize or path to STEP file
            material: Material properties
            loads: List of loads to apply
            constraints: List of boundary conditions to apply
            mesh_size: Target mesh element size (mm)
            element_type: Element type (solver-dependent)
            **kwargs: Additional keyword arguments for kernel-specific options

        Returns:
            Tuple containing:
            - pv_mesh: PyVista mesh for visualization
            - nodes: Node coordinates as numpy array (N, 3)
            - constraint_mask: Boolean mask for constrained nodes (N,)
            - force_mask: Boolean mask for loaded nodes (N,)
            - force_vectors: Force vectors for loaded nodes (M, 3) where M is number of loaded nodes

        Raises:
            NotImplementedError: If the kernel does not support visualization
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support visualization. "
            "Consider using solve() followed by results.show() instead."
        )
