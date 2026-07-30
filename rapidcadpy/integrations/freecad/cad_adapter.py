"""FreeCAD adapter for backend-neutral live CAD references."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


class FreeCADAdapter:
    """Delegate semantic document/object operations to FreeCAD handles."""

    backend_name = "freecad"

    def __init__(self) -> None:
        from .parameter_adapter import FreeCADParameterAdapter

        self.parameter_adapter = FreeCADParameterAdapter()

    def recompute(self, document: Any) -> None:
        document.recompute()

    def save_document(self, document: Any, path: Optional[str] = None) -> str:
        if path:
            resolved = str(Path(path).expanduser().resolve())
            document.saveAs(resolved)
            return resolved
        document.save()
        return str(getattr(document, "FileName", ""))

    def get_property(self, obj: Any, name: str) -> Any:
        return obj.getPropertyByName(name)

    def set_property(self, obj: Any, name: str, value: Any) -> None:
        if name not in getattr(obj, "PropertiesList", []):
            raise KeyError(
                f"FreeCAD object '{getattr(obj, 'Name', '?')}' has no "
                f"property '{name}'."
            )
        setattr(obj, name, value)

    def get_shape(self, obj: Any) -> Any:
        return getattr(obj, "Shape", None)
