"""Backend-neutral native CAD viewport contracts and backend factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Literal, Type, TypeAlias, cast

from .cad_objects import CadDocument


ViewportName: TypeAlias = Literal[
    "current",
    "isometric",
    "front",
    "rear",
    "left",
    "right",
    "top",
    "bottom",
]

VIEWPORT_NAMES: frozenset[str] = frozenset(
    {
        "current",
        "isometric",
        "front",
        "rear",
        "left",
        "right",
        "top",
        "bottom",
    }
)


def normalize_viewport_name(value: str) -> ViewportName:
    """Normalize and validate a backend-neutral viewport name."""

    normalized = str(value).strip().lower()
    if normalized not in VIEWPORT_NAMES:
        available = ", ".join(sorted(VIEWPORT_NAMES))
        raise ValueError(f"Unknown viewport {value!r}. Available views: {available}.")
    return cast(ViewportName, normalized)


def validate_viewport_size(width: int, height: int) -> tuple[int, int]:
    """Validate screenshot dimensions and return normalized integers."""

    normalized_width = int(width)
    normalized_height = int(height)
    if not 64 <= normalized_width <= 4096:
        raise ValueError("Screenshot width must be between 64 and 4096 pixels.")
    if not 64 <= normalized_height <= 4096:
        raise ValueError("Screenshot height must be between 64 and 4096 pixels.")
    return normalized_width, normalized_height


@dataclass(frozen=True)
class ViewportScreenshot:
    """One raster image captured from a native CAD application's viewport."""

    content: bytes = field(repr=False)
    media_type: str = "image/png"
    width: int = 1024
    height: int = 768
    view: ViewportName = "isometric"


class ViewportBackend(ABC):
    """CAD-specific implementation of native viewport image capture."""

    @abstractmethod
    def capture_screenshot(
        self,
        *,
        document: CadDocument,
        view: ViewportName,
        width: int,
        height: int,
        fit: bool,
    ) -> ViewportScreenshot:
        """Capture the current native CAD viewport as a PNG image."""


class ViewportBackendFactory(ABC):
    """Factory for one CAD application's viewport backend."""

    @abstractmethod
    def create(self, document: CadDocument) -> ViewportBackend:
        """Create a viewport backend for ``document``."""


_FACTORIES: Dict[str, Type[ViewportBackendFactory]] = {}


def register_viewport_backend(
    backend_name: str,
    factory_type: Type[ViewportBackendFactory],
) -> None:
    """Register or replace the viewport factory for a CAD backend."""

    normalized = backend_name.strip().lower()
    if not normalized:
        raise ValueError("backend_name must not be empty.")
    _FACTORIES[normalized] = factory_type


def create_viewport_backend(document: CadDocument) -> ViewportBackend:
    """Resolve the CAD-specific viewport backend for a native document."""

    backend_name = document.backend.strip().lower()
    if backend_name == "freecad" and backend_name not in _FACTORIES:
        from .integrations.freecad.viewport import FreeCADViewportBackendFactory

        register_viewport_backend(backend_name, FreeCADViewportBackendFactory)

    factory_type = _FACTORIES.get(backend_name)
    if factory_type is None:
        raise NotImplementedError(
            f"Viewport screenshots are not available for {document.backend!r}."
        )
    return factory_type().create(document)


__all__ = [
    "VIEWPORT_NAMES",
    "ViewportBackend",
    "ViewportBackendFactory",
    "ViewportName",
    "ViewportScreenshot",
    "create_viewport_backend",
    "normalize_viewport_name",
    "register_viewport_backend",
    "validate_viewport_size",
]
