"""Focused ApplicationDiscovery operations composed by :class:`CadSession`."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .integrations.freecad.worker_connection import FreeCADWorkerClient
from .live_backend import LiveCadBackend
from .modeling import (
    ModelingRequest,
    modeling_request_from_dict,
)
from .operation_result import CadOperationSupport
from .session_service import SessionService


class ApplicationDiscovery(SessionService):
    def setup_backend(
        self, cad_system: str = "freecad", document_name: str = "RapidCADPy"
    ) -> Dict[str, Any]:
        cad_system = cad_system.strip().lower()
        if cad_system not in {"freecad"}:
            return self._error(f"Unsupported backend '{cad_system}'. MVP supports only 'freecad'.")

        self._live_backend = None
        if self.execution_mode == "gui":
            ready = self.launch_freecad_gui()
            if not ready.get("ok"):
                return ready
            result = self._gui_connection.call(
                "setup_backend",
                {"cad_system": cad_system, "document_name": document_name},
            )
            if result.get("ok"):
                result["execution_mode"] = "freecad_gui"
                result["gui_pid"] = self._gui_connection.pid
            return result

        if self._worker is not None:
            self._worker.close()
            self._worker = None

        previous_logging_disable = logging.root.manager.disable
        try:
            if self._app_factory is not None:
                freecad_lib_path = None
                self.app = self._app_factory(document_name)
            else:
                if os.environ.get("RAPIDCADPY_CAD_WORKER") != "1":
                    logging.disable(logging.CRITICAL)
                try:
                    from rapidcadpy.integrations.freecad.app import (
                        FreeCADApp,
                        ensure_freecad_python_path,
                    )
                finally:
                    logging.disable(previous_logging_disable)

                freecad_lib_path = ensure_freecad_python_path()
                self.app = FreeCADApp(doc_name=document_name)
        except Exception as exc:
            logging.disable(previous_logging_disable)
            if os.environ.get("RAPIDCADPY_CAD_WORKER") == "1":
                return self._error(
                    "Could not initialize FreeCAD backend inside worker. "
                    f"Error: {type(exc).__name__}: {exc}"
                )
            worker = FreeCADWorkerClient()
            worker_result = worker.call(
                "setup_backend",
                {"cad_system": cad_system, "document_name": document_name},
            )
            if not worker_result.get("ok"):
                worker.close()
                return self._error(
                    "Could not initialize FreeCAD backend directly or through "
                    "FreeCAD worker. Direct error: "
                    f"{type(exc).__name__}: {exc}. Worker error: "
                    f"{worker_result.get('error')}"
                )
            self._worker = worker
            self.backend_name = cad_system
            worker_result["execution_mode"] = "freecad_worker"
            return worker_result

        self.backend_name = cad_system
        self.active_workplane = None
        self.active_workplane_id = None
        self.active_shape_id = None
        self.objects.clear()
        self.runtime_objects.clear()
        self._shape_wrappers.clear()
        self.profile_references.clear()
        self.path_references.clear()
        self._profile_cache.clear()
        self.operations.clear()
        self.parameters.clear()
        self.geometry_signatures.clear()
        self.document.clear()
        self.cad_document = getattr(self.app, "cad_document", None)
        self._recalculate_document_revision()
        self._counters.clear()

        op_id = self._record(
            "setup_backend",
            {"cad_system": cad_system, "document_name": document_name},
            [],
            f"Initialized {cad_system} document '{document_name}'",
        )
        return self._ok(
            summary=f"Initialized FreeCAD backend with document '{document_name}'",
            operation_id=op_id,
            backend=self.backend_name,
            freecad_lib_path=freecad_lib_path,
            document_revision=self.document_revision,
        )

    def launch_freecad_gui(self) -> Dict[str, Any]:
        """Launch one bridge-enabled FreeCAD GUI and retain its connection."""
        if self._gui_connection is not None:
            ping = self._gui_connection.call("ping", {})
            if ping.get("ok"):
                self.backend_name = "freecad"
                self.active_cad_software = "freecad"
                self.active_target_id = self._gui_connection.instance_id
                return self._ok(
                    summary="FreeCAD GUI already connected",
                    execution_mode="freecad_gui",
                    gui_pid=self._gui_connection.pid,
                    bridge_host=self._gui_connection.host,
                    bridge_port=self._gui_connection.port,
                    software="freecad",
                    target_id=self._gui_connection.instance_id,
                    capabilities=list(self._gui_connection.capabilities),
                )
            self._gui_connection = None
            self._worker = None
        try:
            from rapidcadpy.integrations.freecad.gui_connection import (
                FreeCADGuiConnection,
            )

            connection = FreeCADGuiConnection.launch()
        except Exception as exc:
            return self._error(f"Could not launch FreeCAD GUI: {type(exc).__name__}: {exc}")
        self._live_backend = None
        self._gui_connection = connection
        self._worker = connection
        self.backend_name = "freecad"
        self.active_cad_software = "freecad"
        self.active_target_id = connection.instance_id
        return self._ok(
            summary="Launched and connected to FreeCAD GUI",
            execution_mode="freecad_gui",
            gui_pid=connection.pid,
            bridge_host=connection.host,
            bridge_port=connection.port,
            software="freecad",
            target_id=connection.instance_id,
            capabilities=list(self._gui_connection.capabilities),
        )

    def list_cad_applications(
        self,
        software: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Discover applications through registered live integration factories."""
        names = (software,) if software else self._live_backend_registry.supported_software
        applications = []
        unreachable = []
        try:
            for name in names:
                factory = self._live_backend_registry.factory(name)
                for application in factory.discover():
                    destination = applications if application.connected else unreachable
                    destination.append(application.to_dict())
        except Exception as exc:
            return self._error(f"Could not discover CAD applications: {exc}")
        return self._ok(
            summary=f"Found {len(applications)} attachable CAD application(s)",
            applications=applications,
            application_count=len(applications),
            discovered_count=len(applications) + len(unreachable),
            unreachable_applications=unreachable,
            unreachable_count=len(unreachable),
            supported_software=list(self._live_backend_registry.supported_software),
            active_target_id=self.active_target_id,
        )

    def select_cad_application(
        self,
        software: str,
        target_id: Optional[str] = None,
        use_active_document: bool = True,
    ) -> Dict[str, Any]:
        """Select a compatible family of live services from one integration."""
        if self.execution_mode == "embedded":
            return self._error("Cannot attach from inside an embedded CAD process.")
        backend = None
        try:
            factory = self._live_backend_registry.factory(software)
            backend = factory.create(target_id)
            ping = backend.connection.call("ping", {})
            if not ping.get("ok"):
                backend.connection.close()
                return ping
            result = {"ok": True, "active_document": ping.get("active_document")}
            if use_active_document:
                hydrated = backend.hydration.hydrate_active_document()
                if hydrated.get("ok"):
                    result.update(hydrated)
                else:
                    result["warning"] = hydrated.get("error", "Could not hydrate active document.")
        except Exception as exc:
            if backend is not None:
                backend.connection.close()
            return self._error(f"Could not select CAD application: {exc}")

        previous = self._live_backend
        self._live_backend = backend
        # Existing fluent session calls already delegate through this transport.
        self._worker = backend.connection
        self._gui_connection = None
        self.app = None
        self.cad_document = None
        self.active_workplane = None
        self.active_workplane_id = None
        self.active_shape_id = None
        self.objects.clear()
        self.runtime_objects.clear()
        self._shape_wrappers.clear()
        self.profile_references.clear()
        self.path_references.clear()
        self._profile_cache.clear()
        self.parameters.clear()
        self.geometry_signatures.clear()
        self.drawings.clear()
        self.current_drawing_id = None
        self.operations.clear()
        self._counters.clear()
        self.document = dict(result.get("document", {}))
        self.document_revision = result.get("document_revision")
        application = backend.application
        self.backend_name = application.software
        self.active_cad_software = application.software
        self.active_target_id = application.target_id
        if previous is not None and previous.connection is not backend.connection:
            try:
                previous.connection.close()
            except Exception as exc:
                result.setdefault("warnings", []).append(
                    f"Previous connection cleanup failed: {exc}"
                )
        result.update(application.to_dict())
        result["summary"] = f"Selected {application.display_name} as the active CAD application"
        return result

    @property
    def live_backend(self) -> Optional[LiveCadBackend]:
        """Selected service family for runtime composition, never tool serialization."""
        return self._live_backend

    def inspect_modeling_support(
        self, definition: ModelingRequest | dict[str, Any]
    ) -> Dict[str, Any]:
        """Inspect the new modeling contract without mutating the document."""
        if isinstance(definition, dict):
            try:
                definition = modeling_request_from_dict(definition)
            except (TypeError, ValueError, KeyError) as exc:
                return self._error(
                    f"Invalid modeling request: {exc}", error_code="invalid_cad_request"
                )
        if self._live_backend is None:
            if self._worker is not None:
                return self._worker.call(
                    "inspect_modeling_support", {"definition": definition.to_dict()}
                )
            factory = getattr(
                getattr(self.cad_document, "adapter", None), "profile_operations", None
            )
            if callable(factory):
                support = factory(self).inspect_support(definition)
            else:
                support = CadOperationSupport(
                    definition.kind,
                    "unsupported",
                    "none",
                    "The active backend does not implement persistent modeling.",
                )
            return self._ok(summary=support.reason, support=support.to_dict())
        try:
            support = self._live_backend.capabilities.inspect_support(definition)
        except Exception as exc:
            return self._error(f"Could not inspect modeling support: {exc}")
        return self._ok(summary=support.reason, support=support.to_dict())

    def apply_modeling(
        self, definition: ModelingRequest | dict[str, Any], *, expected_revision: str
    ) -> Dict[str, Any]:
        """Apply shared intent through the selected live modeling service."""
        if isinstance(definition, dict):
            try:
                definition = modeling_request_from_dict(definition)
            except (TypeError, ValueError, KeyError) as exc:
                return self._error(
                    f"Invalid modeling request: {exc}", error_code="invalid_cad_request"
                )
        if not isinstance(expected_revision, str) or not expected_revision.strip():
            return self._error("expected_revision must be a non-empty document revision.")
        if self._live_backend is None:
            if self._worker is not None:
                return self._worker.call(
                    "apply_modeling",
                    {
                        "definition": definition.to_dict(),
                        "expected_revision": expected_revision,
                    },
                )
            factory = getattr(
                getattr(self.cad_document, "adapter", None), "profile_operations", None
            )
            if not callable(factory):
                return self._operation_error(
                    "Selected backend cannot apply this modeling definition.",
                    error_code="cad_operation_not_supported",
                )
            try:
                return factory(self).apply_modeling(definition, expected_revision)
            except (TypeError, ValueError, NotImplementedError) as exc:
                return self._error(
                    f"Invalid modeling request: {exc}", error_code="invalid_cad_request"
                )
        try:
            result = self._live_backend.modeling.apply(
                definition, expected_revision=expected_revision
            )
        except Exception as exc:
            return self._error(f"Could not apply modeling definition: {exc}")
        if result.ok:
            self.document_revision = result.document_revision
        return result.to_dict()

    def get_active_cad_application(self) -> Dict[str, Any]:
        """Describe the currently selected backend-neutral CAD target."""
        if self._live_backend is not None:
            ping = self._live_backend.connection.call("ping", {})
            if not ping.get("ok"):
                return self._error(
                    "The active CAD application is no longer reachable: "
                    f"{ping.get('error', 'unknown error')}"
                )
            return self._ok(
                summary=f"{self._live_backend.application.display_name} is the active CAD application",
                selected=True,
                **self._live_backend.application.to_dict(),
                active_document=ping.get("active_document"),
            )
        if self._gui_connection is None or self.active_target_id is None:
            return self._ok(
                summary="No active CAD application is selected",
                selected=False,
                supported_software=list(self._live_backend_registry.supported_software),
            )
        ping = self._gui_connection.call("ping", {})
        if not ping.get("ok"):
            return self._error(
                "The active CAD application is no longer reachable: "
                f"{ping.get('error', 'unknown error')}"
            )
        return self._ok(
            summary="FreeCAD is the active CAD application",
            selected=True,
            software=self.active_cad_software,
            display_name="FreeCAD",
            target_id=self.active_target_id,
            instance_id=self._gui_connection.instance_id,
            pid=self._gui_connection.pid,
            active_document=ping.get("active_document"),
            capabilities=list(self._gui_connection.capabilities),
            bridge_transport=ping.get("bridge_transport"),
        )

    def install_cad_connector(
        self,
        software: str,
        install_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Install the connector for one supported CAD application."""
        normalized = software.strip().lower()
        if normalized != "freecad":
            return self._error(f"Unsupported CAD software '{software}'. Available: freecad.")
        result = self.install_freecad_connector(mod_dir=install_dir)
        if result.get("ok"):
            result.update(
                {
                    "software": "freecad",
                    "display_name": "FreeCAD",
                }
            )
        return result

    def list_freecad_instances(self) -> Dict[str, Any]:
        """List running FreeCAD GUIs that advertise a RapidCADPy bridge."""
        try:
            from rapidcadpy.integrations.freecad.gui_connection import (
                FreeCADGuiConnection,
            )
            from rapidcadpy.integrations.freecad.instance_registry import (
                instance_registry_dir,
            )

            discovered = FreeCADGuiConnection.list_instances()
            registry_dir = instance_registry_dir()
        except Exception as exc:
            return self._error(f"Could not discover FreeCAD GUIs: {type(exc).__name__}: {exc}")
        instances = [item for item in discovered if item.get("connected")]
        unreachable = [item for item in discovered if not item.get("connected")]
        summary = f"Found {len(instances)} attachable FreeCAD GUI instance(s)"
        if unreachable:
            summary += f"; discovered {len(unreachable)} additional unreachable instance(s)"
        if not discovered:
            summary += f"; checked discovery directory {registry_dir}"
        return self._ok(
            summary=summary,
            instances=instances,
            instance_count=len(instances),
            discovered_count=len(discovered),
            unreachable_instances=unreachable,
            unreachable_count=len(unreachable),
            registry_dir=str(registry_dir),
        )

    def install_freecad_connector(
        self,
        mod_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Install RapidCADPy's auto-start bridge into FreeCAD's user modules."""
        try:
            from rapidcadpy.integrations.freecad.connector_addon import (
                install_freecad_connector,
            )

            return install_freecad_connector(mod_dir=mod_dir)
        except Exception as exc:
            return self._error(f"Could not install FreeCAD connector: {type(exc).__name__}: {exc}")

    def attach_freecad(
        self,
        instance_id: Optional[str] = None,
        use_active_document: bool = True,
    ) -> Dict[str, Any]:
        """Attach to an addon-enabled FreeCAD GUI and optionally hydrate its document."""
        if self.execution_mode == "embedded":
            return self._error("Cannot attach from inside the FreeCAD GUI process.")
        try:
            from rapidcadpy.integrations.freecad.gui_connection import (
                FreeCADGuiConnection,
            )

            connection = FreeCADGuiConnection.attach(instance_id=instance_id)
            ping = connection.call("ping", {})
            if not ping.get("ok"):
                return ping
        except Exception as exc:
            return self._error(f"Could not attach to FreeCAD GUI: {exc}")

        self._live_backend = None
        self._gui_connection = connection
        self._worker = connection
        self.backend_name = "freecad"
        self.active_cad_software = "freecad"
        self.active_target_id = connection.instance_id
        attached = self._ok(
            summary="Attached to running FreeCAD GUI",
            execution_mode="freecad_gui_attach",
            instance_id=connection.instance_id,
            gui_pid=connection.pid,
            bridge_host=connection.host,
            bridge_port=connection.port,
            software="freecad",
            target_id=connection.instance_id,
            capabilities=list(self._gui_connection.capabilities),
            active_document=ping.get("active_document"),
            freecad_version=ping.get("freecad_version"),
        )
        if not use_active_document:
            return attached

        hydrated = connection.call("use_active_document", {})
        if not hydrated.get("ok"):
            attached["warning"] = hydrated.get("error")
            return attached
        hydrated.update(
            {
                "summary": ("Attached to running FreeCAD GUI and hydrated its active document"),
                "execution_mode": "freecad_gui_attach",
                "instance_id": connection.instance_id,
                "gui_pid": connection.pid,
                "software": "freecad",
                "display_name": "FreeCAD",
                "target_id": connection.instance_id,
                "capabilities": list(self._gui_connection.capabilities),
            }
        )
        return hydrated
