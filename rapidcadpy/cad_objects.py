"""Backend-neutral references to live native CAD documents and objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Protocol


class CadAdapter(Protocol):
    """Operations required by live CAD object references.

    Adapters operate on native handles. The semantic objects in this module do
    not copy or replace the native CAD model.
    """

    backend_name: str

    def recompute(self, document: Any) -> None: ...

    def save_document(self, document: Any, path: Optional[str] = None) -> str: ...

    def get_property(self, obj: Any, name: str) -> Any: ...

    def set_property(self, obj: Any, name: str, value: Any) -> None: ...

    def get_shape(self, obj: Any) -> Any: ...


@dataclass
class CadDocument:
    """Reference to one live native CAD document."""

    backend: str
    native_handle: Any = field(repr=False)
    adapter: CadAdapter = field(repr=False)
    name: str = ""
    label: str = ""
    file_name: str = ""
    revision: Optional[str] = None
    _object_ids_by_native_name: Dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _native_names_by_object_id: Dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _parameter_ids_by_name: Dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _next_native_object_id: int = field(
        default=0,
        init=False,
        repr=False,
        compare=False,
    )
    _next_parameter_id: int = field(
        default=0,
        init=False,
        repr=False,
        compare=False,
    )

    def object_id(
        self,
        native_name: str,
        id_factory: Optional[Callable[[], str]] = None,
    ) -> str:
        """Return the stable RapidCAD ID for one native document object."""
        normalized_name = str(native_name).strip()
        if not normalized_name:
            raise ValueError("native_name must not be empty.")
        existing = self._object_ids_by_native_name.get(normalized_name)
        if existing is not None:
            return existing
        if id_factory is None:
            object_id = self._next_available_object_id()
        else:
            object_id = str(id_factory())
            while object_id in self._native_names_by_object_id:
                object_id = str(id_factory())
        return self.bind_object_id(normalized_name, object_id)

    def bind_object_id(self, native_name: str, object_id: str) -> str:
        """Bind a public ID to a native name, preserving a one-to-one mapping.

        A shape wrapper may move from an extrusion to a dependent boolean or
        fillet feature while retaining its public RapidCAD identity. In that
        case the previous native feature is deliberately released from the ID
        and receives a new one if it is encountered during later hydration.
        """
        normalized_name = str(native_name).strip()
        normalized_id = str(object_id).strip()
        if not normalized_name:
            raise ValueError("native_name must not be empty.")
        if not normalized_id:
            raise ValueError("object_id must not be empty.")

        previous_id = self._object_ids_by_native_name.get(normalized_name)
        if previous_id is not None and previous_id != normalized_id:
            self._native_names_by_object_id.pop(previous_id, None)

        previous_name = self._native_names_by_object_id.get(normalized_id)
        if previous_name is not None and previous_name != normalized_name:
            self._object_ids_by_native_name.pop(previous_name, None)

        self._object_ids_by_native_name[normalized_name] = normalized_id
        self._native_names_by_object_id[normalized_id] = normalized_name
        return normalized_id

    def parameter_id(
        self,
        name: str,
        id_factory: Optional[Callable[[], str]] = None,
    ) -> str:
        """Return the stable RapidCAD ID for one named document parameter."""
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("Parameter name must not be empty.")
        existing = self._parameter_ids_by_name.get(normalized_name)
        if existing is not None:
            return existing
        if id_factory is None:
            self._next_parameter_id += 1
            parameter_id = f"parameter_{self._next_parameter_id}"
        else:
            parameter_id = str(id_factory())
            while parameter_id in self._parameter_ids_by_name.values():
                parameter_id = str(id_factory())
        self._parameter_ids_by_name[normalized_name] = parameter_id
        return parameter_id

    def _next_available_object_id(self) -> str:
        while True:
            self._next_native_object_id += 1
            object_id = f"native_{self._next_native_object_id}"
            if object_id not in self._native_names_by_object_id:
                return object_id

    def recompute(self) -> None:
        """Recompute the native document through its backend adapter."""
        self.adapter.recompute(self.native_handle)

    def save(self, path: Optional[str] = None) -> str:
        """Save the native document and return the resulting path."""
        saved_path = self.adapter.save_document(self.native_handle, path)
        if saved_path:
            self.file_name = saved_path
        return saved_path

    def to_dict(self) -> Dict[str, Any]:
        """Return the serializable document identity."""
        return {
            "backend": self.backend,
            "name": self.name,
            "label": self.label,
            "file_name": self.file_name,
            "revision": self.revision,
        }


@dataclass
class CadObject:
    """Backend-neutral reference to one live native CAD object."""

    id: str
    document: CadDocument
    native_handle: Any = field(repr=False)
    native_name: str = ""
    native_type: str = ""
    label: str = ""
    semantic_type: str = "native_object"
    visibility: Optional[bool] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    geometry: Dict[str, Any] = field(default_factory=dict)
    capabilities: FrozenSet[str] = field(default_factory=frozenset)

    @property
    def backend(self) -> str:
        return self.document.backend

    @property
    def is_parametric(self) -> bool:
        return False

    @property
    def shape(self) -> Any:
        """Return the live native result shape, when the object has one."""
        return self.document.adapter.get_shape(self.native_handle)

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def get_property(self, name: str) -> Any:
        """Read a property from the live native object."""
        return self.document.adapter.get_property(self.native_handle, name)

    def set_property(self, name: str, value: Any, recompute: bool = True) -> None:
        """Write a native property without replacing the native feature."""
        if "set_properties" not in self.capabilities:
            raise NotImplementedError(
                f"{self.native_type or self.semantic_type} does not advertise "
                "the 'set_properties' capability."
            )
        self.document.adapter.set_property(self.native_handle, name, value)
        # The serialized metadata and revision describe the last hydrated
        # snapshot. Invalidate them rather than presenting stale values.
        self.properties.pop(name, None)
        self.document.revision = None
        if recompute:
            self.document.recompute()

    def to_dict(self) -> Dict[str, Any]:
        """Return the serializable semantic view; never expose native handles."""
        result: Dict[str, Any] = {
            "id": self.id,
            "type": self.semantic_type,
            "label": self.label,
            "native_name": self.native_name,
            "native_type": self.native_type,
            "backend": self.backend,
            "visibility": self.visibility,
            "properties": self.properties,
            "dependencies": {
                "depends_on": self.depends_on,
                "dependents": self.dependents,
            },
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "capabilities": sorted(self.capabilities),
        }
        if self.geometry:
            result["geometry"] = self.geometry
        return result


@dataclass
class CadFeature(CadObject):
    """A live native object that produces or modifies model geometry."""

    semantic_type: str = "feature"

    @property
    def is_parametric(self) -> bool:
        return self.supports("feature_history")


@dataclass(frozen=True)
class ParameterBinding:
    """One native feature property driven by a named parameter expression."""

    object_id: str
    property_name: str
    expression: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "object_id": self.object_id,
            "property_name": self.property_name,
            "expression": self.expression,
        }


@dataclass
class CadParameter:
    """Backend-neutral reference to one persistent named CAD parameter."""

    id: str
    document: CadDocument
    native_handle: Any = field(repr=False)
    name: str = ""
    label: str = ""
    parameter_type: str = "number"
    value: Any = None
    unit: Optional[str] = None
    expression: Optional[str] = None
    evaluated_value: Any = None
    source: str = "native_user_parameter"
    writable: bool = True
    renamable: bool = False
    deletable: bool = True
    bindable: bool = True
    dependencies: List[str] = field(default_factory=list)
    dependents: List[ParameterBinding] = field(default_factory=list)

    @property
    def backend(self) -> str:
        return self.document.backend

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable parameter snapshot without its native handle."""
        return {
            "id": self.id,
            "name": self.name,
            "label": self.label,
            "parameter_type": self.parameter_type,
            "value": self.value,
            "unit": self.unit,
            "expression": self.expression,
            "evaluated_value": self.evaluated_value,
            "source": self.source,
            "backend": self.backend,
            "writable": self.writable,
            "renamable": self.renamable,
            "deletable": self.deletable,
            "bindable": self.bindable,
            "dependencies": list(self.dependencies),
            "dependents": [binding.to_dict() for binding in self.dependents],
        }
