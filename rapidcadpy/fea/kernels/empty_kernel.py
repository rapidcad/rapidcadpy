"""Solver-less placeholder kernel.

This is useful when FEA visualization should stay importable even when no
actual solver backend (for example ``torchfem``) is available.
"""

from __future__ import annotations

from .base import FEAKernel


class EmptyKernel(FEAKernel):
    """Kernel placeholder that intentionally does not solve or optimize."""

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def get_solver_name(cls) -> str:
        return "visualization-only"

    def solve(
        self,
        shape,
        material,
        loads,
        constraints,
        mesh_size,
        element_type,
        verbose: bool = False,
    ):
        raise NotImplementedError(
            "EmptyKernel does not support solve(). Install a real FEA backend."
        )

    def optimize(
        self,
        shape,
        material,
        loads,
        constraints,
        mesh_size,
        element_type,
        volume_fraction: float = 0.5,
        num_iterations: int = 100,
        penalization: float = 3.0,
        filter_radius: float = 0.0,
        move_limit: float = 0.2,
        rho_min: float = 1e-3,
        use_autograd: bool = False,
        verbose: bool = False,
    ):
        raise NotImplementedError(
            "EmptyKernel does not support optimize(). Install a real FEA backend."
        )
