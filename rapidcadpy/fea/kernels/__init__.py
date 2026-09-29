"""Finite-element analysis kernel interfaces and implementations."""

from .base import FEAKernel
from .empty_kernel import EmptyKernel

__all__ = ["EmptyKernel", "FEAKernel"]
