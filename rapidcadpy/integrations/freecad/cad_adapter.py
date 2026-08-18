"""FreeCAD adapter for backend-neutral live CAD references."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from ...cad_objects import CadDocument, CadFeature, CadObject
from .errors import FreeCADNativeFeatureError


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

    def create_boolean_feature(
        self,
        document: CadDocument,
        target: CadObject,
        tools: Iterable[CadObject],
        *,
        type_id: str,
        prefix: str,
        result_id: Optional[str] = None,
    ) -> CadFeature:
        """Create one history-preserving native FreeCAD boolean feature.

        This is the sole FreeCAD-kernel implementation for parametric cut and
        union operations.  Callers receive a backend-neutral feature reference;
        native handles never leave the adapter boundary.
        """

        tool_list = list(tools)
        if not tool_list:
            raise FreeCADNativeFeatureError("A boolean operation needs at least one tool.")
        if any(item.document is not document for item in tool_list):
            raise FreeCADNativeFeatureError(
                "Native FreeCAD booleans require all operands in the same document."
            )

        native_document = document.native_handle
        native_feature = None
        try:
            native_feature = native_document.addObject(
                type_id, self._next_name(native_document, prefix)
            )
            if type_id == "Part::Cut":
                native_feature.Base = target.native_handle
                native_feature.Tool = tool_list[0].native_handle
            elif type_id == "Part::MultiFuse":
                native_feature.Shapes = [
                    target.native_handle,
                    *(item.native_handle for item in tool_list),
                ]
            else:
                raise FreeCADNativeFeatureError(f"Unsupported FreeCAD boolean: {type_id}")
            document.recompute()
            shape = getattr(native_feature, "Shape", None)
            if shape is None or shape.isNull():
                raise ValueError(f"FreeCAD recomputed an empty {type_id} shape.")
            self.set_boolean_result_visibility(
                native_feature,
                [target.native_handle, *(item.native_handle for item in tool_list)],
            )
        except Exception as exc:
            if native_feature is not None:
                try:
                    native_document.removeObject(native_feature.Name)
                    document.recompute()
                except Exception:
                    pass
            if isinstance(exc, FreeCADNativeFeatureError):
                raise
            raise FreeCADNativeFeatureError(
                f"Could not create native {type_id}; no direct-geometry fallback was applied."
            ) from exc

        public_id = document.bind_object_id(
            str(native_feature.Name), result_id or target.id
        )
        return CadFeature(
            id=public_id,
            document=document,
            native_handle=native_feature,
            native_name=str(native_feature.Name),
            native_type=str(native_feature.TypeId),
            label=str(getattr(native_feature, "Label", native_feature.Name)),
            capabilities=frozenset(
                {"geometry", "get_properties", "set_properties", "feature_history"}
            ),
        )

    @staticmethod
    def set_boolean_result_visibility(result: Any, operands: Iterable[Any]) -> None:
        """Show only the terminal boolean result while preserving its operands."""

        for operand in operands:
            view_object = getattr(operand, "ViewObject", None)
            if view_object is not None:
                view_object.Visibility = False
        result_view = getattr(result, "ViewObject", None)
        if result_view is not None:
            result_view.Visibility = True

    @staticmethod
    def _next_name(document: Any, prefix: str) -> str:
        index = 1
        while document.getObject(f"{prefix}_{index}") is not None:
            index += 1
        return f"{prefix}_{index}"
