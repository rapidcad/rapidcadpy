"""Live FreeCAD factory using the existing GUI connector, without kernel imports."""

from __future__ import annotations

from typing import Any

from ...bridge_contract import advertised_capabilities
from ...live_backend import (
    CadApplication,
    CapabilityInspector,
    DocumentHydrator,
    LiveCadBackendFactory,
    LiveCadConnection,
    ModelingBackend,
)
from ...modeling import (
    ModelingRequest,
)
from ...operation_result import CadOperationResult, CadOperationSupport


class FreeCADModelingBackend:
    """Dispatch supported planar definitions through the selected connection."""

    def __init__(self, connection: LiveCadConnection) -> None:
        self._connection = connection

    def inspect_support(self, definition: ModelingRequest) -> CadOperationSupport:
        from .spline_geometry import definition_support

        support = definition_support(definition)
        if support.status == "unsupported":
            return support
        result = self._connection.call(
            "inspect_modeling_support", {"definition": definition.to_dict()}
        )
        if result.get("ok") and result.get("support"):
            return CadOperationSupport(**result["support"])
        return CadOperationSupport(
            support.operation,
            "unsupported",
            "none",
            result.get(
                "error", "The connector cannot inspect persistent modeling support."
            ),
        )

    def inspect_capabilities(self) -> dict[str, Any]:
        return self._connection.call("inspect_capabilities", {})

    def apply(
        self, definition: ModelingRequest, *, expected_revision: str
    ) -> CadOperationResult:
        from .spline_geometry import definition_support

        support = definition_support(definition)
        if support.status == "unsupported":
            return CadOperationResult.failure(
                support.reason,
                error_code="cad_operation_not_supported",
                support=support,
            )
        return CadOperationResult.from_dict(
            self._connection.call(
                "apply_modeling",
                {
                    "definition": definition.to_dict(),
                    "expected_revision": expected_revision,
                },
            )
        )


class FreeCADDocumentHydrator:
    def __init__(self, connection: LiveCadConnection) -> None:
        self._connection = connection

    def hydrate_active_document(self) -> dict[str, Any]:
        return self._connection.call("use_active_document", {})


class FreeCADLiveBackendFactory(LiveCadBackendFactory):
    @staticmethod
    def _application(
        target_id: str,
        *,
        connected: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> CadApplication:
        # Preserve the existing legacy capabilities while that API is migrated.
        from .capabilities import FREECAD_CAPABILITIES

        return CadApplication(
            software="freecad",
            display_name="FreeCAD",
            target_id=target_id,
            connected=connected,
            capabilities=advertised_capabilities(FREECAD_CAPABILITIES, (metadata or {}).get("bridge_contract")),
            metadata=metadata or {},
        )

    def discover(self) -> tuple[CadApplication, ...]:
        from .gui_connection import FreeCADGuiConnection

        return tuple(
            self._application(
                item["instance_id"],
                connected=bool(item.get("connected")),
                metadata=item,
            )
            for item in FreeCADGuiConnection.list_instances()
        )

    def connect(
        self, target_id: str | None = None
    ) -> tuple[CadApplication, LiveCadConnection]:
        from .gui_connection import FreeCADGuiConnection

        connection = FreeCADGuiConnection.attach(instance_id=target_id)
        application = self._application(
            connection.instance_id,
            metadata={
                "bridge_contract": getattr(connection, "bridge_contract", None),
                "instance_id": connection.instance_id,
                "pid": connection.pid,
                "gui_pid": connection.pid,
                "bridge_host": connection.host,
                "bridge_port": connection.port,
                "execution_mode": "freecad_gui_attach",
            },
        )
        return application, connection

    def create_modeling(self, connection: LiveCadConnection) -> ModelingBackend:
        return FreeCADModelingBackend(connection)

    def create_hydrator(self, connection: LiveCadConnection) -> DocumentHydrator:
        return FreeCADDocumentHydrator(connection)

    def create_capability_inspector(
        self, connection: LiveCadConnection
    ) -> CapabilityInspector:
        return FreeCADModelingBackend(connection)
