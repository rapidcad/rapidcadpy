"""Stable contracts for applying semantic features through CAD adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from .cad_objects import CadFeature, CadObject
from .feature import Feature


@dataclass(frozen=True)
class FeatureSupport:
    operation: str
    status: str
    mode: str
    reason: str


@dataclass(frozen=True)
class FeatureResult:
    feature: CadFeature
    definition: Feature
    document_revision: Optional[str]
    created_object_ids: tuple[str, ...] = ()
    changed_object_ids: tuple[str, ...] = ()
    removed_object_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class FeatureExecutor(Protocol):
    def inspect_support(self, definition: Feature, target: CadObject) -> FeatureSupport:
        """Report support without mutating the native document."""

    def apply(
        self,
        definition: Feature,
        target: CadObject,
        expected_revision: Optional[str],
    ) -> FeatureResult:
        """Create or update a native feature and return its public reference."""
