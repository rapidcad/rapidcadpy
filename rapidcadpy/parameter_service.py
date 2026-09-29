"""Focused ParameterService operations composed by :class:`CadSession`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .cad_objects import CadObject, CadParameter
from .session_service import SessionService


class ParameterService(SessionService):
    def add_parameter(self, name: str, value: float, units: str = "mm") -> Dict[str, Any]:
        """Compatibility wrapper for the persistent named-parameter API."""
        parameter_type = "angle" if units.strip().lower() in {"deg", "rad"} else "length"
        return self.create_parameter(
            name=name,
            parameter_type=parameter_type,
            value=value,
            unit=units,
        )

    def list_parameters(self) -> Dict[str, Any]:
        """List persistent named parameters in the active native document."""
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call("list_parameters", {})
        parameter_snapshots = [parameter.to_dict() for parameter in self.parameters.values()]
        return self._ok(
            summary=f"{len(parameter_snapshots)} named parameter(s)",
            parameter_count=len(parameter_snapshots),
            parameters=parameter_snapshots,
            document_revision=self.document_revision,
        )

    def get_parameter(self, parameter_id: str) -> Dict[str, Any]:
        """Inspect one persistent named parameter by RapidCAD ID or name."""
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "get_parameter",
                {"parameter_id": parameter_id},
            )
        parameter = self._find_parameter(parameter_id)
        if parameter is None:
            return self._error(f"Unknown parameter_id or name '{parameter_id}'.")
        return self._ok(
            summary=f"Named parameter {parameter.name}",
            parameter=parameter.to_dict(),
            document_revision=self.document_revision,
        )

    def create_parameter(
        self,
        name: str,
        parameter_type: str,
        value: Any,
        unit: Optional[str] = None,
        expression: Optional[str] = None,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a persistent typed named parameter in the native document."""
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "create_parameter",
                {
                    "name": name,
                    "parameter_type": parameter_type,
                    "value": value,
                    "unit": unit,
                    "expression": expression,
                    "expected_revision": expected_revision,
                },
            )
        revision_error = self._parameter_revision_error(expected_revision)
        if revision_error is not None:
            return revision_error
        try:
            native_document, adapter = self._parameter_context()
            available_names = {item.name for item in self.parameters.values()}
            with adapter.transaction(native_document, f"Create parameter {name}"):
                adapter.create_parameter(
                    native_document,
                    name=name,
                    parameter_type=parameter_type,
                    value=value,
                    unit=unit,
                    expression=expression,
                    available_names=available_names,
                )
        except Exception as exc:
            return self._error(f"create_parameter failed: {type(exc).__name__}: {exc}")
        result = self._rehydrate_parameter_document("create_parameter")
        if not result.get("ok"):
            return result
        created = next(
            (
                parameter.to_dict()
                for parameter in self.parameters.values()
                if parameter.name == name
            ),
            None,
        )
        result.update(
            {
                "summary": f"Created named parameter '{name}'",
                "parameter": created,
                "changed_parameters": [created["id"]] if created else [],
            }
        )
        return result

    def set_parameters(
        self,
        updates: list[Dict[str, Any]],
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update named parameters atomically and recompute the native document."""
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "set_parameters",
                {
                    "updates": updates,
                    "expected_revision": expected_revision,
                },
            )
        if not updates:
            return self._error("set_parameters requires at least one update.")
        revision_error = self._parameter_revision_error(expected_revision)
        if revision_error is not None:
            return revision_error

        resolved: list[tuple[CadParameter, Dict[str, Any]]] = []
        for update in updates:
            identifier = str(update.get("parameter_id") or update.get("name") or "").strip()
            parameter = self._find_parameter(identifier)
            if parameter is None:
                return self._error(f"Unknown parameter_id or name '{identifier}'.")
            if "value" not in update and "expression" not in update:
                return self._error(f"Update for '{parameter.name}' requires value or expression.")
            resolved.append((parameter, update))

        try:
            native_document, adapter = self._parameter_context()
            available_names = {item.name for item in self.parameters.values()}
            with adapter.transaction(native_document, "Update named parameters"):
                for parameter, update in resolved:
                    if update.get("expression") is not None:
                        adapter.set_expression(
                            parameter.native_handle,
                            str(update["expression"]),
                            available_names,
                        )
                    else:
                        adapter.set_value(
                            parameter.native_handle,
                            update["value"],
                            unit=update.get("unit", parameter.unit),
                            parameter_type=parameter.parameter_type,
                        )
        except Exception as exc:
            return self._error(f"set_parameters failed: {type(exc).__name__}: {exc}")

        changed_names = [parameter.name for parameter, _ in resolved]
        result = self._rehydrate_parameter_document("set_parameters")
        if not result.get("ok"):
            return result
        changed_parameters = [
            item.to_dict() for item in self.parameters.values() if item.name in changed_names
        ]
        affected_ids = sorted(
            {
                binding["object_id"]
                for parameter in changed_parameters
                for binding in parameter["dependents"]
            }
        )
        result.update(
            {
                "summary": (f"Updated {len(changed_parameters)} named parameter(s)"),
                "changed_parameters": changed_parameters,
                "affected_objects": affected_ids,
            }
        )
        return result

    def bind_parameter(
        self,
        parameter_id: str,
        object_id: str,
        property_name: str,
        expression: Optional[str] = None,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Bind a named parameter expression to a generic native feature property."""
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "bind_parameter",
                {
                    "parameter_id": parameter_id,
                    "object_id": object_id,
                    "property_name": property_name,
                    "expression": expression,
                    "expected_revision": expected_revision,
                },
            )
        revision_error = self._parameter_revision_error(expected_revision)
        if revision_error is not None:
            return revision_error
        parameter = self._find_parameter(parameter_id)
        if parameter is None:
            return self._error(f"Unknown parameter_id or name '{parameter_id}'.")
        target = self.runtime_objects.get(object_id)
        if not isinstance(target, CadObject):
            return self._error(f"Unknown native object_id '{object_id}'.")

        try:
            native_document, adapter = self._parameter_context()
            available_names = {item.name for item in self.parameters.values()}
            with adapter.transaction(
                native_document,
                f"Bind parameter {parameter.name}",
            ):
                native_property = adapter.bind_parameter(
                    parameter.native_handle,
                    target.native_handle,
                    property_name,
                    expression,
                    available_names,
                )
        except Exception as exc:
            return self._error(f"bind_parameter failed: {type(exc).__name__}: {exc}")

        parameter_name = parameter.name
        result = self._rehydrate_parameter_document("bind_parameter")
        if not result.get("ok"):
            return result
        rebound_parameter = next(
            (item.to_dict() for item in self.parameters.values() if item.name == parameter_name),
            None,
        )
        result.update(
            {
                "summary": (f"Bound parameter '{parameter_name}' to {object_id}.{property_name}"),
                "parameter": rebound_parameter,
                "binding": {
                    "object_id": object_id,
                    "property_name": property_name,
                    "native_property": native_property,
                    "expression": expression or parameter_name,
                },
            }
        )
        return result

    def _find_parameter(self, identifier: str) -> Optional[CadParameter]:
        if identifier in self.parameters:
            return self.parameters[identifier]
        return next(
            (parameter for parameter in self.parameters.values() if parameter.name == identifier),
            None,
        )

    def _parameter_context(self) -> tuple[Any, Any]:
        if self.cad_document is None:
            raise RuntimeError("No native CAD document is open.")
        adapter = getattr(self.cad_document.adapter, "parameter_adapter", None)
        if adapter is None:
            raise NotImplementedError(
                f"{self.cad_document.backend} does not support persistent named parameters."
            )
        return self.cad_document.native_handle, adapter

    def _parameter_revision_error(
        self,
        expected_revision: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if expected_revision and expected_revision != self.document_revision:
            return self._error(
                "Document revision mismatch: expected "
                f"{expected_revision}, current {self.document_revision}."
            )
        return None

    def _rehydrate_parameter_document(self, source_tool: str) -> Dict[str, Any]:
        if self.cad_document is None:
            return self._error("No native CAD document is open.")
        native_document = self.cad_document.native_handle
        file_name = str(getattr(native_document, "FileName", "")).strip()
        file_path = Path(file_name).expanduser().resolve() if file_name else None
        return self._hydrate_freecad_document(
            native_document,
            file_path,
            source_tool=source_tool,
        )

    def _recalculate_document_revision(self) -> str:
        return self.mutation_coordinator.revision()
