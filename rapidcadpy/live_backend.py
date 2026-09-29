"""Abstract factory for live CAD applications, separate from artifact execution.

A factory supplies a compatible family of discovery, connection, modeling,
hydration and capability services. Connections execute in the selected CAD GUI;
native handles remain on that side of the boundary.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .modeling import ModelingRequest
from .operation_result import CadOperationResult, CadOperationSupport


@dataclass(frozen=True)
class CadApplication:
    software: str
    display_name: str
    target_id: str
    connected: bool = True
    capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Copy transport metadata and reject process-local/native values.
        object.__setattr__(
            self, "metadata", json.loads(json.dumps(self.metadata, allow_nan=False))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "software": self.software,
            "display_name": self.display_name,
            "target_id": self.target_id,
            "connected": self.connected,
            "capabilities": list(self.capabilities),
        }


class LiveCadConnection(Protocol):
    """Transport for the existing generic live-session operation surface."""

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None:
        """Release this connection without terminating the user's application."""
        ...


class ModelingBackend(Protocol):
    def apply(
        self, definition: ModelingRequest, *, expected_revision: str
    ) -> CadOperationResult:
        """Validate revision, transact, recompute, validate and rehydrate.

        Return changed/created/removed IDs, the new revision and support details.
        Failures must roll back native and runtime state. Never silently bake.
        Profile/path creation returns a CadProfile/CadPath dictionary in data.
        """
        ...


class DocumentHydrator(Protocol):
    def hydrate_active_document(self) -> dict[str, Any]:
        """Return the public document snapshot, IDs and revision, without handles."""
        ...


class CapabilityInspector(Protocol):
    def inspect_capabilities(self) -> dict[str, Any]:
        """Report operation support and integration restrictions without native handles."""
        ...

    def inspect_support(self, definition: ModelingRequest) -> CadOperationSupport:
        """Inspect request-level support without changing the document."""
        ...


@dataclass(frozen=True)
class LiveCadBackend:
    application: CadApplication
    connection: LiveCadConnection
    modeling: ModelingBackend
    hydration: DocumentHydrator
    capabilities: CapabilityInspector


class LiveCadBackendFactory(ABC):
    """One implementation per CAD application, instantiated only on demand."""

    @abstractmethod
    def discover(self) -> tuple[CadApplication, ...]: ...

    @abstractmethod
    def connect(
        self, target_id: str | None = None
    ) -> tuple[CadApplication, LiveCadConnection]: ...

    @abstractmethod
    def create_modeling(self, connection: LiveCadConnection) -> ModelingBackend: ...

    @abstractmethod
    def create_hydrator(self, connection: LiveCadConnection) -> DocumentHydrator: ...

    @abstractmethod
    def create_capability_inspector(
        self, connection: LiveCadConnection
    ) -> CapabilityInspector: ...

    def create(self, target_id: str | None = None) -> LiveCadBackend:
        application, connection = self.connect(target_id)
        try:
            return LiveCadBackend(
                application=application,
                connection=connection,
                modeling=self.create_modeling(connection),
                hydration=self.create_hydrator(connection),
                capabilities=self.create_capability_inspector(connection),
            )
        except Exception:
            connection.close()
            raise


def _freecad_factory() -> LiveCadBackendFactory:
    from .integrations.freecad.live_backend import FreeCADLiveBackendFactory

    return FreeCADLiveBackendFactory()


class LiveCadBackendRegistry:
    """Session-injectable registry; registration never imports a CAD kernel."""

    def __init__(self, *, include_defaults: bool = True) -> None:
        self._factories: dict[str, Callable[[], LiveCadBackendFactory]] = {}
        if include_defaults:
            self.register("freecad", _freecad_factory)

    def register(
        self, software: str, factory: Callable[[], LiveCadBackendFactory]
    ) -> None:
        name = software.strip().lower()
        if not name:
            raise ValueError("software must not be empty.")
        if name in self._factories:
            raise ValueError(f"A live factory is already registered for {name!r}.")
        self._factories[name] = factory

    @property
    def supported_software(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def factory(self, software: str) -> LiveCadBackendFactory:
        name = software.strip().lower()
        if name not in self._factories:
            raise ValueError(
                f"Unsupported CAD software {software!r}. Available: "
                + ", ".join(self.supported_software)
            )
        return self._factories[name]()
