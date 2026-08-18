"""Stateful RapidCADPy session for local and live CAD applications."""

from __future__ import annotations

import os
import hashlib
import io
import json
import logging
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cad_objects import CadDocument, CadFeature, CadObject, CadParameter
from .feature import Feature
from .integrations.freecad.worker_connection import (
    FreeCADWorkerClient,
)


@dataclass
class SemanticObject:
    id: str
    type: str
    label: str
    source_op: str
    source: str = "cad_operation"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "source_op": self.source_op,
            "source": self.source,
            "confidence": self.confidence,
            **self.metadata,
        }

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return the fields needed to choose an object for later inspection."""
        summary = {
            "id": self.id,
            "type": self.type,
            "label": self.label,
        }
        for key in (
            "native_name",
            "native_type",
            "backend",
            "visibility",
            "capabilities",
            "geometry",
        ):
            if key in self.metadata:
                summary[key] = self.metadata[key]
        return summary


@dataclass
class OperationRecord:
    id: str
    tool: str
    args: Dict[str, Any]
    outputs: list[str]
    summary: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "args": self.args,
            "outputs": self.outputs,
            "summary": self.summary,
            "timestamp": self.timestamp,
        }


class CadSession:
    """Stateful standalone API around one live RapidCADPy CAD document."""

    _FREECAD_CAPABILITIES = (
        "document.create",
        "document.open",
        "document.save",
        "document.hydrate",
        "code.execute",
        "sketch.work_plane",
        "sketch.rectangle",
        "sketch.circle",
        "sketch.line",
        "sketch.arc",
        "feature.box",
        "feature.extrude",
        "feature.loft",
        "feature.boolean.cut",
        "feature.boolean.union",
        "feature.fillet",
        "feature.hole",
        "object.list",
        "object.inspect",
        "object.set_property",
        "parameter.list",
        "parameter.create",
        "parameter.update",
        "parameter.bind",
        "drawing.generate",
        "drawing.list",
        "drawing.select",
        "drawing.inspect",
        "drawing.dimension.feature",
        "drawing.annotation.note",
        "drawing.annotation.leader",
        "drawing.item.move",
        "drawing.export",
        "viewport.select",
        "viewport.fit",
    )

    def __init__(self, execution_mode: str = "headless") -> None:
        if execution_mode not in {"headless", "gui", "embedded"}:
            raise ValueError("execution_mode must be 'headless', 'gui', or 'embedded'.")
        self.execution_mode = execution_mode
        self._worker: Any = None
        self._gui_connection: Any = None
        self.backend_name: Optional[str] = None
        self.app: Any = None
        self.active_workplane: Any = None
        self.active_workplane_id: Optional[str] = None
        self.active_shape_id: Optional[str] = None
        self.objects: Dict[str, SemanticObject] = {}
        self.runtime_objects: Dict[str, Any] = {}
        # Backend-specific fluent objects are useful for isolated geometry work,
        # but never constitute the public live-CAD object registry.
        self._shape_wrappers: Dict[str, Any] = {}
        self.operations: list[OperationRecord] = []
        self.parameters: Dict[str, CadParameter] = {}
        self.geometry_signatures: Dict[str, Dict[str, Any]] = {}
        self.drawings: Dict[str, Dict[str, Any]] = {}
        self.current_drawing_id: Optional[str] = None
        self.document: Dict[str, Any] = {}
        self.cad_document: Optional[CadDocument] = None
        self.document_revision: Optional[str] = None
        self._counters: Dict[str, int] = {}
        self.active_cad_software: Optional[str] = None
        self.active_target_id: Optional[str] = None

    def setup_backend(
        self, cad_system: str = "freecad", document_name: str = "RapidCADPy"
    ) -> Dict[str, Any]:
        cad_system = cad_system.strip().lower()
        if cad_system not in {"freecad"}:
            return self._error(
                f"Unsupported backend '{cad_system}'. MVP supports only 'freecad'."
            )

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
        self.operations.clear()
        self.parameters.clear()
        self.geometry_signatures.clear()
        self.document.clear()
        self.cad_document = getattr(self.app, "cad_document", None)
        self.document_revision = None
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
                    capabilities=list(self._FREECAD_CAPABILITIES),
                )
            self._gui_connection = None
            self._worker = None
        try:
            from rapidcadpy.integrations.freecad.gui_connection import (
                FreeCADGuiConnection,
            )

            connection = FreeCADGuiConnection.launch()
        except Exception as exc:
            return self._error(
                f"Could not launch FreeCAD GUI: {type(exc).__name__}: {exc}"
            )
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
            capabilities=list(self._FREECAD_CAPABILITIES),
        )

    def list_cad_applications(
        self,
        software: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List running CAD applications through backend-neutral metadata."""
        normalized = software.strip().lower() if software else None
        if normalized not in {None, "freecad"}:
            return self._error(
                f"Unsupported CAD software '{software}'. Available: freecad."
            )
        discovered = self.list_freecad_instances()
        if not discovered.get("ok"):
            return discovered

        applications = [
            {
                **item,
                "target_id": item["instance_id"],
                "software": "freecad",
                "display_name": "FreeCAD",
                "capabilities": list(self._FREECAD_CAPABILITIES),
            }
            for item in discovered["instances"]
        ]
        unreachable = [
            {
                **item,
                "target_id": item["instance_id"],
                "software": "freecad",
                "display_name": "FreeCAD",
                "capabilities": list(self._FREECAD_CAPABILITIES),
            }
            for item in discovered["unreachable_instances"]
        ]
        return self._ok(
            summary=f"Found {len(applications)} attachable CAD application(s)",
            applications=applications,
            application_count=len(applications),
            discovered_count=discovered["discovered_count"],
            unreachable_applications=unreachable,
            unreachable_count=len(unreachable),
            supported_software=["freecad"],
            active_target_id=self.active_target_id,
        )

    def select_cad_application(
        self,
        software: str,
        target_id: Optional[str] = None,
        use_active_document: bool = True,
    ) -> Dict[str, Any]:
        """Select one running CAD application as the active generic target."""
        normalized = software.strip().lower()
        if normalized != "freecad":
            return self._error(
                f"Unsupported CAD software '{software}'. Available: freecad."
            )
        selected = self.attach_freecad(
            instance_id=target_id,
            use_active_document=use_active_document,
        )
        if not selected.get("ok"):
            return selected
        selected.update(
            {
                "summary": "Selected FreeCAD as the active CAD application",
                "software": "freecad",
                "display_name": "FreeCAD",
                "target_id": self.active_target_id,
                "capabilities": list(self._FREECAD_CAPABILITIES),
            }
        )
        return selected

    def get_active_cad_application(self) -> Dict[str, Any]:
        """Describe the currently selected backend-neutral CAD target."""
        if self._gui_connection is None or self.active_target_id is None:
            return self._ok(
                summary="No active CAD application is selected",
                selected=False,
                supported_software=["freecad"],
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
            capabilities=list(self._FREECAD_CAPABILITIES),
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
            return self._error(
                f"Unsupported CAD software '{software}'. Available: freecad."
            )
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
            return self._error(
                f"Could not discover FreeCAD GUIs: {type(exc).__name__}: {exc}"
            )
        instances = [item for item in discovered if item.get("connected")]
        unreachable = [item for item in discovered if not item.get("connected")]
        summary = f"Found {len(instances)} attachable FreeCAD GUI instance(s)"
        if unreachable:
            summary += (
                f"; discovered {len(unreachable)} additional unreachable instance(s)"
            )
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
            return self._error(
                f"Could not install FreeCAD connector: {type(exc).__name__}: {exc}"
            )

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
            capabilities=list(self._FREECAD_CAPABILITIES),
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
                "summary": (
                    "Attached to running FreeCAD GUI and hydrated its active document"
                ),
                "execution_mode": "freecad_gui_attach",
                "instance_id": connection.instance_id,
                "gui_pid": connection.pid,
                "software": "freecad",
                "display_name": "FreeCAD",
                "target_id": connection.instance_id,
                "capabilities": list(self._FREECAD_CAPABILITIES),
            }
        )
        return hydrated

    def use_active_document(self) -> Dict[str, Any]:
        """Hydrate the document currently active in an attached FreeCAD GUI."""
        if self._worker is not None:
            result = self._worker.call("use_active_document", {})
            if result.get("ok") and self._gui_connection is not None:
                result["execution_mode"] = "freecad_gui_attach"
                result["instance_id"] = self._gui_connection.instance_id
                result["gui_pid"] = self._gui_connection.pid
                result["software"] = "freecad"
                result["display_name"] = "FreeCAD"
                result["target_id"] = self._gui_connection.instance_id
                result["capabilities"] = list(self._FREECAD_CAPABILITIES)
            return result
        if self.execution_mode == "gui":
            attached = self.attach_freecad(use_active_document=False)
            if not attached.get("ok"):
                return attached
            return self.use_active_document()

        try:
            from rapidcadpy.integrations.freecad.app import FreeCADApp

            import FreeCAD as App

            document = App.ActiveDocument
            if document is None:
                return self._error("FreeCAD has no active document.")
            if self.app is None:
                self.app = FreeCADApp.from_document(document)
            else:
                try:
                    current_document = self.app.get_doc()
                except (ReferenceError, RuntimeError):
                    current_document = None
                if current_document is not document:
                    if hasattr(self.app, "bind_document"):
                        self.app.bind_document(document)
                    else:
                        self.app = FreeCADApp.from_document(document)
            self.backend_name = "freecad"
            document.recompute()
            file_name = str(getattr(document, "FileName", "")).strip()
            file_path = Path(file_name).expanduser().resolve() if file_name else None
            return self._hydrate_freecad_document(
                document,
                file_path,
                source_tool="use_active_document",
            )
        except Exception as exc:
            return self._error(
                f"use_active_document failed: {type(exc).__name__}: {exc}"
            )

    def new_document(self, name: str = "RapidCADPy") -> Dict[str, Any]:
        ready = self._ensure_gui_session(require_document=False)
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call("new_document", {"name": name})
        if self.app is None and self.execution_mode != "embedded":
            return self._error(
                "No backend configured. Call setup_backend('freecad') first."
            )
        if self.backend_name not in {None, "freecad"}:
            return self._error("new_document MVP supports only FreeCAD backend.")

        try:
            from rapidcadpy.integrations.freecad.app import FreeCADApp

            self.app = FreeCADApp(doc_name=name)
            self.backend_name = "freecad"
        except Exception as exc:
            return self._error(
                f"Could not create document: {type(exc).__name__}: {exc}"
            )

        self.active_workplane = None
        self.active_workplane_id = None
        self.active_shape_id = None
        self.objects.clear()
        self.runtime_objects.clear()
        self._shape_wrappers.clear()
        self.geometry_signatures.clear()
        self.document = {
            "name": getattr(self.app.get_doc(), "Name", name),
            "label": getattr(self.app.get_doc(), "Label", name),
            "file_name": getattr(self.app.get_doc(), "FileName", ""),
        }
        self.cad_document = getattr(self.app, "cad_document", None)
        self.document_revision = None
        op_id = self._record(
            "new_document", {"name": name}, [], f"Created document '{name}'"
        )
        return self._ok(summary=f"Created document '{name}'", operation_id=op_id)

    def open_document(self, path: str) -> Dict[str, Any]:
        """Open and hydrate a native FreeCAD document into the live session."""
        if self._worker is not None:
            result = self._worker.call("open_document", {"path": path})
            if result.get("ok") and self._gui_connection is not None:
                result["execution_mode"] = "freecad_gui"
                result["gui_pid"] = self._gui_connection.pid
            return result
        if self.execution_mode == "gui":
            ready = self.launch_freecad_gui()
            if not ready.get("ok"):
                return ready
            return self.open_document(path)

        file_path = Path(path).expanduser().resolve()
        if not file_path.is_file():
            return self._error(f"FreeCAD file not found: {file_path}")

        if self.app is None or self.backend_name != "freecad":
            setup = self.setup_backend("freecad", file_path.stem or "RapidCADPy")
            if not setup.get("ok"):
                return setup
            if self._worker is not None:
                return self._worker.call("open_document", {"path": str(file_path)})

        try:
            from rapidcadpy.integrations.freecad.app import ensure_freecad_python_path

            ensure_freecad_python_path()
            import FreeCAD as App

            old_doc = self.app.get_doc()
            try:
                old_file_name = getattr(old_doc, "FileName", "")
                old_path = Path(old_file_name).resolve() if old_file_name else None
                old_name = old_doc.Name
            except ReferenceError:
                old_doc = None
                old_path = None
                old_name = None
            if old_path == file_path:
                doc = old_doc
            else:
                # A FreeCADApp always owns a document. Close that scratch document
                # before opening the requested file to avoid internal-name clashes.
                if old_name is not None:
                    App.closeDocument(old_name)
                doc = App.openDocument(str(file_path))
                if hasattr(self.app, "bind_document"):
                    self.app.bind_document(doc)
                else:
                    self.app._fc_doc = doc
            doc.recompute()
        except Exception as exc:
            return self._error(f"open_document failed: {type(exc).__name__}: {exc}")

        try:
            return self._hydrate_freecad_document(doc, file_path)
        except Exception as exc:
            return self._error(
                f"Could not hydrate opened document: {type(exc).__name__}: {exc}"
            )

    def execute_code(
        self, code: str, allow_direct_geometry: bool = False
    ) -> Dict[str, Any]:
        """Execute code in the GUI, rejecting new baked features by default."""
        ready = self._ensure_gui_session(require_document=False)
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "execute_code",
                {
                    "code": code,
                    "allow_direct_geometry": allow_direct_geometry,
                },
            )
        if self.execution_mode != "embedded" and self.backend_name != "freecad":
            return self._error(
                "execute_code requires an attached or embedded FreeCAD session."
            )

        normalized = code.strip()
        if not normalized:
            return self._error("FreeCAD code must not be empty.")

        execution_dir = Path(
            os.environ.get(
                "RAPIDCADPY_LIVE_CODE_DIR",
                str(Path(tempfile.gettempdir()) / "rapidcadpy-live-code"),
            )
        ).expanduser()
        execution_dir.mkdir(parents=True, exist_ok=True)
        existing_files = {
            path.resolve() for path in execution_dir.iterdir() if path.is_file()
        }
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        namespace: Dict[str, Any] = {
            "__name__": "__rapidcadpy_live__",
            "__builtins__": __builtins__,
        }
        import FreeCAD as App

        flat_feature_types = {"Part::Feature", "PartDesign::Feature"}

        def flat_feature_snapshot() -> Dict[tuple[str, str], tuple[str, int]]:
            snapshot: Dict[tuple[str, str], tuple[str, int]] = {}
            for document_name, document in App.listDocuments().items():
                for obj in document.Objects:
                    type_id = str(getattr(obj, "TypeId", ""))
                    if type_id not in flat_feature_types:
                        continue
                    shape = getattr(obj, "Shape", None)
                    if shape is None or shape.isNull():
                        continue
                    try:
                        shape_hash = int(shape.hashCode())
                    except Exception:
                        shape_hash = hash(str(shape))
                    snapshot[(str(document_name), str(obj.Name))] = (
                        type_id,
                        shape_hash,
                    )
            return snapshot

        original_documents = dict(App.listDocuments())
        original_object_names = {
            document_name: {str(obj.Name) for obj in document.Objects}
            for document_name, document in original_documents.items()
        }
        original_flat_shapes = {}
        for document_name, document in original_documents.items():
            for obj in document.Objects:
                if str(getattr(obj, "TypeId", "")) not in flat_feature_types:
                    continue
                shape = getattr(obj, "Shape", None)
                if shape is not None and not shape.isNull():
                    original_flat_shapes[(document_name, str(obj.Name))] = shape.copy()
        original_active_name = str(
            getattr(getattr(App, "ActiveDocument", None), "Name", "")
        )
        before_flat_features = flat_feature_snapshot()
        transaction_documents = []
        for document in original_documents.values():
            try:
                document.openTransaction("RapidCADPy live code")
                transaction_documents.append(document)
            except Exception:
                continue

        def rollback_execution() -> None:
            current_documents = dict(App.listDocuments())
            for document in transaction_documents:
                if str(getattr(document, "Name", "")) in current_documents:
                    try:
                        document.abortTransaction()
                    except Exception:
                        pass
            for document_name, document in original_documents.items():
                if document_name not in current_documents:
                    continue
                known_names = original_object_names[document_name]
                for obj in reversed(list(document.Objects)):
                    if str(obj.Name) not in known_names:
                        try:
                            document.removeObject(str(obj.Name))
                        except Exception:
                            pass
                for (
                    shape_document,
                    object_name,
                ), shape in original_flat_shapes.items():
                    if shape_document != document_name:
                        continue
                    obj = document.getObject(object_name)
                    if obj is not None:
                        try:
                            obj.Shape = shape.copy()
                        except Exception:
                            pass
                try:
                    document.recompute()
                except Exception:
                    pass
            for document_name in set(current_documents) - set(original_documents):
                try:
                    App.closeDocument(document_name)
                except Exception:
                    pass
            if original_active_name and original_active_name in App.listDocuments():
                try:
                    App.setActiveDocument(original_active_name)
                except Exception:
                    pass

        def commit_execution() -> None:
            current_documents = App.listDocuments()
            for document in transaction_documents:
                if str(getattr(document, "Name", "")) in current_documents:
                    try:
                        document.commitTransaction()
                    except Exception:
                        pass

        previous_cwd = Path.cwd()
        try:
            os.chdir(execution_dir)
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(compile(normalized, "<rapidcadpy-live>", "exec"), namespace)
        except Exception as exc:
            rollback_execution()
            return self._error(
                "Live FreeCAD code failed: "
                f"{type(exc).__name__}: {exc}\n"
                f"stdout:\n{stdout_buffer.getvalue()}\n"
                f"stderr:\n{stderr_buffer.getvalue()}"
            )
        finally:
            os.chdir(previous_cwd)

        after_flat_features = flat_feature_snapshot()
        changed_flat_features = sorted(
            key
            for key, signature in after_flat_features.items()
            if before_flat_features.get(key) != signature
        )
        if changed_flat_features and not allow_direct_geometry:
            rollback_execution()
            formatted = ", ".join(
                f"{document_name}.{object_name}"
                for document_name, object_name in changed_flat_features
            )
            return self._error(
                "Live FreeCAD code created or modified baked Part::Feature "
                f"geometry ({formatted}). The transaction was rolled back. Use "
                "native Sketcher/Part/PartDesign features, or explicitly set "
                "allow_direct_geometry=True when history loss is intentional."
            )

        commit_execution()

        hydrated = self.use_active_document()
        if not hydrated.get("ok"):
            return self._error(
                "Live FreeCAD code completed but its active document could not "
                f"be hydrated: {hydrated.get('error', 'unknown error')}"
            )
        generated_files = sorted(
            str(path.resolve())
            for path in execution_dir.iterdir()
            if path.is_file() and path.resolve() not in existing_files
        )
        hydrated.update(
            {
                "summary": "Executed code visibly in the attached FreeCAD GUI",
                "execution_target": "attached_freecad_gui",
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
                "generated_files": generated_files,
                "warnings": (
                    [
                        "Direct geometry mode was explicitly enabled; baked "
                        "Part::Feature objects may not preserve construction history."
                    ]
                    if allow_direct_geometry
                    else []
                ),
            }
        )
        return hydrated

    def work_plane(
        self, plane: str = "XY", offset: Optional[float] = None
    ) -> Dict[str, Any]:
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call("work_plane", {"plane": plane, "offset": offset})
        ready = self._require_app()
        if ready is not None:
            return ready
        try:
            wp = self.app.work_plane(plane, offset=offset)
        except Exception as exc:
            return self._error(
                f"Could not create workplane: {type(exc).__name__}: {exc}"
            )

        wp_id = self._new_id("workplane")
        op_id = self._record(
            "work_plane",
            {"plane": plane, "offset": offset},
            [wp_id],
            f"Created {plane.upper()} workplane",
        )
        self.active_workplane = wp
        self.active_workplane_id = wp_id
        self.runtime_objects[wp_id] = wp
        self.objects[wp_id] = SemanticObject(
            id=wp_id,
            type="workplane",
            label=f"{plane.upper()} workplane",
            source_op=op_id,
            metadata={"plane": plane.upper(), "offset": offset},
        )
        return self._ok(
            summary=f"Created {plane.upper()} workplane",
            operation_id=op_id,
            object_id=wp_id,
            active_workplane_id=wp_id,
        )

    def move_to(self, x: float, y: float) -> Dict[str, Any]:
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call("move_to", {"x": x, "y": y})
        ready = self._require_workplane()
        if ready is not None:
            return ready
        try:
            self.active_workplane.move_to(float(x), float(y))
        except Exception as exc:
            return self._error(f"move_to failed: {type(exc).__name__}: {exc}")
        op_id = self._record(
            "move_to", {"x": x, "y": y}, [], f"Moved sketch cursor to ({x}, {y})"
        )
        return self._ok(
            summary=f"Moved sketch cursor to ({x}, {y})", operation_id=op_id
        )

    def line_to(self, x: float, y: float) -> Dict[str, Any]:
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call("line_to", {"x": x, "y": y})
        ready = self._require_workplane()
        if ready is not None:
            return ready
        try:
            self.active_workplane.line_to(float(x), float(y))
        except Exception as exc:
            return self._error(f"line_to failed: {type(exc).__name__}: {exc}")
        op_id = self._record(
            "line_to", {"x": x, "y": y}, [], f"Drew line to ({x}, {y})"
        )
        return self._ok(summary=f"Drew line to ({x}, {y})", operation_id=op_id)

    def three_point_arc(
        self,
        mid_x: float,
        mid_y: float,
        end_x: float,
        end_y: float,
    ) -> Dict[str, Any]:
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "three_point_arc",
                {
                    "mid_x": mid_x,
                    "mid_y": mid_y,
                    "end_x": end_x,
                    "end_y": end_y,
                },
            )
        ready = self._require_workplane()
        if ready is not None:
            return ready
        try:
            self.active_workplane.three_point_arc(
                (float(mid_x), float(mid_y)),
                (float(end_x), float(end_y)),
            )
        except Exception as exc:
            return self._error(f"three_point_arc failed: {type(exc).__name__}: {exc}")
        op_id = self._record(
            "three_point_arc",
            {
                "mid_x": mid_x,
                "mid_y": mid_y,
                "end_x": end_x,
                "end_y": end_y,
            },
            [],
            f"Drew arc through ({mid_x}, {mid_y}) to ({end_x}, {end_y})",
        )
        return self._ok(
            summary=f"Drew arc through ({mid_x}, {mid_y}) to ({end_x}, {end_y})",
            operation_id=op_id,
        )

    def rect(
        self, width: float, height: float, centered: bool = True
    ) -> Dict[str, Any]:
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "rect", {"width": width, "height": height, "centered": centered}
            )
        ready = self._require_workplane()
        if ready is not None:
            return ready
        try:
            self.active_workplane.rect(
                float(width), float(height), centered=bool(centered)
            )
        except Exception as exc:
            return self._error(f"rect failed: {type(exc).__name__}: {exc}")

        profile_id = self._new_id("profile")
        op_id = self._record(
            "rect",
            {"width": width, "height": height, "centered": centered},
            [profile_id],
            f"Drew rectangle {width} x {height}",
        )
        self.objects[profile_id] = SemanticObject(
            id=profile_id,
            type="profile",
            label="rectangle",
            source_op=op_id,
            metadata={
                "width": float(width),
                "height": float(height),
                "centered": bool(centered),
            },
        )
        return self._ok(
            summary=f"Drew rectangle {width} x {height}",
            operation_id=op_id,
            object_id=profile_id,
            active_workplane_id=self.active_workplane_id,
        )

    def circle(self, radius: float) -> Dict[str, Any]:
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call("circle", {"radius": radius})
        ready = self._require_workplane()
        if ready is not None:
            return ready
        try:
            self.active_workplane.circle(float(radius))
        except Exception as exc:
            return self._error(f"circle failed: {type(exc).__name__}: {exc}")

        profile_id = self._new_id("profile")
        op_id = self._record(
            "circle", {"radius": radius}, [profile_id], f"Drew circle radius {radius}"
        )
        self.objects[profile_id] = SemanticObject(
            id=profile_id,
            type="profile",
            label="circle",
            source_op=op_id,
            metadata={"radius": float(radius)},
        )
        return self._ok(
            summary=f"Drew circle radius {radius}",
            operation_id=op_id,
            object_id=profile_id,
            active_workplane_id=self.active_workplane_id,
        )

    def box(
        self,
        length: float,
        width: float,
        height: float,
        centered: bool = True,
    ) -> Dict[str, Any]:
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "box",
                {
                    "length": length,
                    "width": width,
                    "height": height,
                    "centered": centered,
                },
            )
        ready = self._require_workplane()
        if ready is not None:
            return ready
        try:
            shape = self.active_workplane.box(
                float(length),
                float(width),
                float(height),
                centered=bool(centered),
            )
        except Exception as exc:
            return self._error(f"box failed: {type(exc).__name__}: {exc}")

        shape_id = self._new_id("shape")
        if getattr(shape, "feature", None) is not None:
            shape_id = shape.feature.document.bind_object_id(
                shape.feature.native_name,
                shape_id,
            )
            shape.feature.id = shape_id
            self.runtime_objects[shape_id] = shape.feature
            self._shape_wrappers[shape_id] = shape
        else:
            return self._error("box did not produce a live native CAD feature.")
        self.active_shape_id = shape_id
        signature = self._geometry_signature(shape)
        self.geometry_signatures[shape_id] = signature
        op_id = self._record(
            "box",
            {
                "length": length,
                "width": width,
                "height": height,
                "centered": centered,
            },
            [shape_id],
            f"Created native box {length} x {width} x {height}",
        )
        self.objects[shape_id] = SemanticObject(
            id=shape_id,
            type="solid",
            label="box",
            source_op=op_id,
            metadata={
                "length": float(length),
                "width": float(width),
                "height": float(height),
                "centered": bool(centered),
                "geometry": signature,
            },
        )
        return self._ok(
            summary=f"Created native box {length} x {width} x {height}",
            operation_id=op_id,
            object_id=shape_id,
            active_shape_id=shape_id,
            geometry=signature,
        )

    def extrude(
        self, distance: float, operation: str = "new_body", symmetric: bool = False
    ) -> Dict[str, Any]:
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "extrude",
                {"distance": distance, "operation": operation, "symmetric": symmetric},
            )
        ready = self._require_workplane()
        if ready is not None:
            return ready
        op_map = {
            "new_body": "NewBodyFeatureOperation",
            "join": "JoinBodyFeatureOperation",
            "cut": "Cut",
            "intersect": "Intersect",
        }
        backend_operation = op_map.get(operation, operation)
        try:
            shape = self.active_workplane.extrude(
                float(distance), operation=backend_operation, symmetric=bool(symmetric)
            )
        except Exception as exc:
            return self._error(f"extrude failed: {type(exc).__name__}: {exc}")

        shape_id = self._new_id("shape")
        if getattr(shape, "feature", None) is not None:
            shape_id = shape.feature.document.bind_object_id(
                shape.feature.native_name,
                shape_id,
            )
            shape.feature.id = shape_id
            self.runtime_objects[shape_id] = shape.feature
            self._shape_wrappers[shape_id] = shape
        else:
            return self._error("extrude did not produce a live native CAD feature.")
        self.active_shape_id = shape_id
        signature = self._geometry_signature(shape)
        self.geometry_signatures[shape_id] = signature
        op_id = self._record(
            "extrude",
            {"distance": distance, "operation": operation, "symmetric": symmetric},
            [shape_id],
            f"Extruded active profile by {distance}",
        )
        self.objects[shape_id] = SemanticObject(
            id=shape_id,
            type="solid",
            label="extruded_solid",
            source_op=op_id,
            metadata={
                "distance": float(distance),
                "operation": operation,
                "geometry": signature,
            },
        )
        return self._ok(
            summary=f"Extruded active profile by {distance}",
            operation_id=op_id,
            object_id=shape_id,
            active_shape_id=shape_id,
            geometry=signature,
        )

    def loft(
        self,
        profile_workplane_ids: List[str],
        make_solid: bool = True,
        ruled: bool = False,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a native loft through ordered, existing workplane profiles."""

        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "loft",
                {
                    "profile_workplane_ids": profile_workplane_ids,
                    "make_solid": make_solid,
                    "ruled": ruled,
                    "expected_revision": expected_revision,
                },
            )
        ready = self._require_app()
        if ready is not None:
            return ready
        if expected_revision and expected_revision != self.document_revision:
            return self._error(
                "Document revision mismatch: expected "
                f"{expected_revision}, current {self.document_revision}."
            )
        if len(profile_workplane_ids) < 2:
            return self._error(
                "loft requires at least two ordered profile_workplane_ids."
            )
        if len(set(profile_workplane_ids)) != len(profile_workplane_ids):
            return self._error("loft profile_workplane_ids must be distinct.")

        workplanes = []
        for workplane_id in profile_workplane_ids:
            semantic_object = self.objects.get(workplane_id)
            workplane = self.runtime_objects.get(workplane_id)
            if semantic_object is None or semantic_object.type != "workplane":
                return self._error(
                    f"{workplane_id!r} is not a known workplane ID."
                )
            if workplane is None:
                return self._error(
                    f"Workplane {workplane_id!r} has no live native handle."
                )
            workplanes.append(workplane)

        if not callable(getattr(workplanes[0], "loft", None)):
            return self._error(
                f"Workplane {profile_workplane_ids[0]!r} cannot create a native loft."
            )

        try:
            shape = workplanes[0].loft(
                workplanes[1:],
                make_solid=bool(make_solid),
                ruled=bool(ruled),
            )
        except Exception as exc:
            return self._error(f"loft failed: {type(exc).__name__}: {exc}")

        shape_id = self._new_id("shape")
        if getattr(shape, "feature", None) is not None:
            shape_id = shape.feature.document.bind_object_id(
                shape.feature.native_name,
                shape_id,
            )
            shape.feature.id = shape_id
            self.runtime_objects[shape_id] = shape.feature
            self._shape_wrappers[shape_id] = shape
        else:
            return self._error("loft did not produce a live native CAD feature.")
        self.active_shape_id = shape_id
        signature = self._geometry_signature(shape)
        self.geometry_signatures[shape_id] = signature
        op_id = self._record(
            "loft",
            {
                "profile_workplane_ids": list(profile_workplane_ids),
                "make_solid": bool(make_solid),
                "ruled": bool(ruled),
            },
            [shape_id],
            f"Lofted {len(workplanes)} workplane profiles",
        )
        self.objects[shape_id] = SemanticObject(
            id=shape_id,
            type="solid",
            label="lofted_solid",
            source_op=op_id,
            metadata={
                "profile_workplane_ids": list(profile_workplane_ids),
                "make_solid": bool(make_solid),
                "ruled": bool(ruled),
                "geometry": signature,
            },
        )
        return self._ok(
            summary=f"Lofted {len(workplanes)} workplane profiles",
            operation_id=op_id,
            object_id=shape_id,
            active_shape_id=shape_id,
            document_revision=self.document_revision,
            geometry=signature,
        )

    def cut(self, target_id: str, tool_id: str) -> Dict[str, Any]:
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "cut", {"target_id": target_id, "tool_id": tool_id}
            )
        target = self.runtime_objects.get(target_id)
        tool = self.runtime_objects.get(tool_id)
        if not isinstance(target, CadObject):
            return self._error(f"Object '{target_id}' is not a live native CAD object.")
        if not isinstance(tool, CadObject):
            return self._error(f"Object '{tool_id}' is not a live native CAD object.")
        try:
            result = target.document.adapter.create_boolean_feature(
                target.document,
                target,
                [tool],
                type_id="Part::Cut",
                prefix="Cut",
                result_id=target_id,
            )
        except Exception as exc:
            return self._error(f"cut failed: {type(exc).__name__}: {exc}")

        result_id = target_id
        self.runtime_objects[result_id] = result
        wrapper = self._shape_wrappers.get(result_id)
        if wrapper is not None:
            wrapper.bind_native(result.document, result)
            wrapper.obj = result.shape
        self.active_shape_id = result_id
        signature = self._geometry_signature(result)
        self.geometry_signatures[result_id] = signature
        if result_id in self.objects:
            self.objects[result_id].metadata["geometry"] = signature
            self.objects[result_id].metadata["last_boolean"] = {
                "operation": "cut",
                "tool_id": tool_id,
            }
        op_id = self._record(
            "cut",
            {"target_id": target_id, "tool_id": tool_id},
            [result_id],
            f"Cut {tool_id} from {target_id}",
        )
        return self._ok(
            summary=f"Cut {tool_id} from {target_id}",
            operation_id=op_id,
            object_id=result_id,
            active_shape_id=result_id,
            geometry=signature,
        )

    def union(
        self,
        target_id: str,
        tool_ids: List[str],
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                {
                    "target_id": target_id,
                    "tool_ids": tool_ids,
                    "expected_revision": expected_revision,
                }
            )
        if expected_revision and expected_revision != self.document_revision:
            return self._error(
                "Document revision mismatch: expected "
                f"{expected_revision}, current {self.document_revision}."
            )
        target = self.runtime_objects.get(target_id)
        if not isinstance(target, CadObject):
            return self._error(f"Object '{target_id}' is not a live native CAD object.")
        if not tool_ids:
            return self._error("tool_ids must contain at least one shape ID.")
        tools = []
        for tool_id in tool_ids:
            tool = self.runtime_objects.get(tool_id)
            if not isinstance(tool, CadObject):
                return self._error(f"Object '{tool_id}' is not a live native CAD object.")
            tools.append(tool)
        try:
            result = target.document.adapter.create_boolean_feature(
                target.document,
                target,
                tools,
                type_id="Part::MultiFuse",
                prefix="Fuse",
                result_id=target_id,
            )
        except Exception as exc:
            return self._error(f"union failed: {type(exc).__name__}: {exc}")

        self.runtime_objects[target_id] = result
        wrapper = self._shape_wrappers.get(target_id)
        if wrapper is not None:
            wrapper.bind_native(result.document, result)
            wrapper.obj = result.shape
        self.active_shape_id = target_id
        signature = self._geometry_signature(result)
        self.geometry_signatures[target_id] = signature
        if target_id in self.objects:
            self.objects[target_id].metadata["geometry"] = signature
            self.objects[target_id].metadata["last_boolean"] = {
                "operation": "union",
                "tool_ids": list(tool_ids),
            }
        op_id = self._record(
            "union",
            {"target_id": target_id, "tool_ids": list(tool_ids)},
            [target_id],
            f"United {target_id} with {len(tool_ids)} shape(s)",
        )
        return self._ok(
            summary=f"United {target_id} with {len(tool_ids)} shape(s)",
            operation_id=op_id,
            object_id=target_id,
            active_shape_id=target_id,
            geometry=signature,
        )

    def fillet(
        self,
        shape_id: Optional[str] = None,
        radius: float = 1.0,
        selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "fillet",
                {"shape_id": shape_id, "radius": radius, "selector": selector},
            )
        shape_id = shape_id or self.active_shape_id
        if not shape_id:
            return self._error("No shape_id provided and no active shape exists.")
        shape = self._shape_wrappers.get(shape_id)
        if shape is None:
            return self._error(
                f"Object '{shape_id}' has no fluent geometry wrapper; use a native feature operation."
            )
        try:
            if selector:
                result = shape.edges(selector).fillet(float(radius))
            else:
                result = shape.fillet(float(radius))
        except Exception as exc:
            return self._error(f"fillet failed: {type(exc).__name__}: {exc}")

        if not isinstance(getattr(result, "feature", None), CadObject):
            return self._error("fillet did not produce a live native CAD feature.")
        self.runtime_objects[shape_id] = result.feature
        self._shape_wrappers[shape_id] = result
        self.active_shape_id = shape_id
        signature = self._geometry_signature(result)
        self.geometry_signatures[shape_id] = signature
        if shape_id in self.objects:
            self.objects[shape_id].metadata["geometry"] = signature
            self.objects[shape_id].metadata["last_fillet"] = {
                "radius": float(radius),
                "selector": selector,
            }
        op_id = self._record(
            "fillet",
            {"shape_id": shape_id, "radius": radius, "selector": selector},
            [shape_id],
            f"Filleted {shape_id} radius {radius}",
        )
        return self._ok(
            summary=f"Filleted {shape_id} radius {radius}",
            operation_id=op_id,
            object_id=shape_id,
            active_shape_id=shape_id,
            geometry=signature,
        )

    def hole(
        self,
        target_id: str,
        diameter: float,
        center: Optional[list[float]] = None,
        axis: Optional[list[float]] = None,
        through: bool = True,
        depth: Optional[float] = None,
        hole_type: str = "simple",
        countersink_diameter: Optional[float] = None,
        countersink_angle: Optional[float] = None,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a native, semantic hole feature on a live CAD object.

        ``center`` is the entry-point centre in model millimetres. A through
        hole extends through the target; a blind hole requires ``depth``.
        """

        ready = self._ensure_gui_session(require_document=True)
        if ready is not None:
            return ready
        params = {
            "target_id": target_id,
            "diameter": diameter,
            "center": center,
            "axis": axis,
            "through": through,
            "depth": depth,
            "hole_type": hole_type,
            "countersink_diameter": countersink_diameter,
            "countersink_angle": countersink_angle,
            "expected_revision": expected_revision,
        }
        if self._worker is not None:
            return self._worker.call("hole", params)
        try:
            center_vector = self._vector3(center or [0.0, 0.0, 0.0], "center")
            axis_vector = self._vector3(axis or [0.0, 0.0, 1.0], "axis")
            from .features import HoleFeature, HoleTermination, HoleType

            definition = HoleFeature(
                target_id=target_id,
                center=center_vector,
                axis=axis_vector,
                diameter_mm=float(diameter),
                termination=(
                    HoleTermination.THROUGH if through else HoleTermination.BLIND
                ),
                depth_mm=None if through else depth,
                hole_type=HoleType(hole_type),
                countersink_diameter_mm=countersink_diameter,
                countersink_angle_degrees=countersink_angle,
            )
        except (TypeError, ValueError) as exc:
            return self._error(f"Invalid hole definition: {exc}")
        return self.apply_feature(definition, target_id, expected_revision)

    def apply_feature(
        self,
        feature: "Feature",
        target_id: str,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply a semantic feature to a hydrated native CAD object.

        This is the generic mutation path for objects that were created outside
        RapidCADPy as well as for objects created through the fluent API.
        """

        ready = self._ensure_gui_session(require_document=True)
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._error(
                "Applying a Feature object through the remote worker is not yet "
                "supported; use the typed live-CAD operation."
            )
        if expected_revision and expected_revision != self.document_revision:
            return self._error(
                "Document revision mismatch: expected "
                f"{expected_revision}, current {self.document_revision}."
            )
        candidate = self.runtime_objects.get(target_id)
        target = candidate if isinstance(candidate, CadObject) else getattr(candidate, "feature", None)
        if not isinstance(target, CadObject):
            return self._error(f"Object '{target_id}' is not a live native CAD object.")
        if self.cad_document is None or self.cad_document.backend != "freecad":
            return self._error("No selected CAD feature executor supports this document.")
        try:
            from .integrations.freecad.feature_executor import FreeCADFeatureExecutor

            result = FreeCADFeatureExecutor().apply(feature, target, expected_revision)
            native_document = self.cad_document.native_handle
            file_name = str(getattr(native_document, "FileName", "")).strip()
            file_path = Path(file_name).expanduser().resolve() if file_name else None
            hydrated = self._hydrate_freecad_document(
                native_document,
                file_path,
                source_tool="apply_feature",
            )
            if not hydrated.get("ok"):
                return hydrated
        except Exception as exc:
            return self._error(f"Feature application failed: {type(exc).__name__}: {exc}")
        return self._ok(
            summary=f"Applied {type(feature).__name__} to {target_id}",
            object_id=result.feature.id,
            feature=feature.to_dict(),
            document_revision=self.document_revision,
            created_object_ids=list(result.created_object_ids),
            changed_object_ids=list(result.changed_object_ids),
            removed_object_ids=list(result.removed_object_ids),
            warnings=list(result.warnings),
        )

    @staticmethod
    def _vector3(value: list[float], name: str) -> tuple[float, float, float]:
        if len(value) != 3:
            raise ValueError(f"{name} must contain exactly three coordinates.")
        return (float(value[0]), float(value[1]), float(value[2]))

    def export_step(self, path: str, shape_id: Optional[str] = None) -> Dict[str, Any]:
        return self._export(path, "step", shape_id=shape_id)

    def export_stl(self, path: str, shape_id: Optional[str] = None) -> Dict[str, Any]:
        return self._export(path, "stl", shape_id=shape_id)

    def export_native(
        self, path: str, shape_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._export(path, "native", shape_id=shape_id)

    def describe_freecad_file(self, path: str) -> Dict[str, Any]:
        if self._worker is not None:
            return self._worker.call("describe_freecad_file", {"path": path})
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            return self._error(f"FreeCAD file not found: {file_path}")

        try:
            from rapidcadpy.integrations.freecad.app import ensure_freecad_python_path

            ensure_freecad_python_path()
            import FreeCAD as App
        except Exception as exc:
            return self._error(
                f"Could not import FreeCAD to describe file: {type(exc).__name__}: {exc}"
            )

        doc = None
        owns_opened_document = False
        try:
            active_doc = self.app.get_doc() if self.app is not None else None
            try:
                active_file_name = getattr(active_doc, "FileName", "")
                active_path = (
                    Path(active_file_name).resolve() if active_file_name else None
                )
            except ReferenceError:
                active_doc = None
                active_path = None
            if active_path == file_path:
                doc = active_doc
            else:
                doc = App.openDocument(str(file_path))
                owns_opened_document = True
            objects = [self._describe_freecad_object(obj) for obj in doc.Objects]
            return self._ok(
                summary=f"Described FreeCAD file with {len(objects)} objects",
                document={
                    "name": getattr(doc, "Name", None),
                    "label": getattr(doc, "Label", None),
                    "file_name": getattr(doc, "FileName", str(file_path)),
                },
                objects=objects,
            )
        except Exception as exc:
            return self._error(
                f"describe_freecad_file failed: {type(exc).__name__}: {exc}"
            )
        finally:
            if doc is not None and owns_opened_document:
                try:
                    App.closeDocument(doc.Name)
                except Exception:
                    pass

    def render(
        self,
        path: str,
        view: str = "iso",
        shape_id: Optional[str] = None,
        width: int = 1000,
        height: int = 800,
    ) -> Dict[str, Any]:
        resolved_path = self._resolve_export_path(path)
        if resolved_path.suffix.lower() != ".png":
            return self._error("Render output path must end in '.png'.")
        if not (64 <= int(width) <= 4096 and 64 <= int(height) <= 4096):
            return self._error("Render width and height must be between 64 and 4096.")

        if self._worker is not None:
            worker_result = self._worker.call(
                "render",
                {
                    "path": str(resolved_path),
                    "view": view,
                    "shape_id": shape_id,
                    "width": width,
                    "height": height,
                },
            )
            if worker_result.get("ok"):
                return worker_result

            # If the worker cannot render, export a temporary STL and retry in
            # an isolated subprocess owned by the caller's Python runtime.
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                temp_stl = Path(tmp.name)
            try:
                export_result = self._worker.call(
                    "export_stl", {"path": str(temp_stl), "shape_id": shape_id}
                )
                if not export_result.get("ok"):
                    return worker_result
                self._ensure_parent(str(resolved_path))
                render_worker = (
                    Path(__file__).resolve().parent / "workers" / "render_stl_worker.py"
                )
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(render_worker),
                        str(temp_stl),
                        str(resolved_path),
                        view,
                        str(int(width)),
                        str(int(height)),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=float(os.environ.get("RAPIDCADPY_RENDER_TIMEOUT", "120")),
                    check=False,
                )
                if completed.returncode != 0:
                    details = (completed.stderr or completed.stdout).strip()
                    return self._error(
                        "render subprocess failed with exit code "
                        f"{completed.returncode}: {details[-2000:]}"
                    )
            except Exception as exc:
                return self._error(
                    f"render failed in worker and server fallback: "
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                temp_stl.unlink(missing_ok=True)
            return self._render_response(
                resolved_path, view, shape_id, int(width), int(height)
            )

        shape_id = shape_id or self.active_shape_id
        if not shape_id:
            return self._error("No shape_id provided and no active shape exists.")
        shape = self.runtime_objects.get(shape_id)
        if shape is None:
            return self._error(f"Unknown shape_id '{shape_id}'")
        try:
            self._ensure_parent(str(resolved_path))
            shape.to_png(
                str(resolved_path), view=view, width=int(width), height=int(height)
            )
        except Exception as exc:
            return self._error(f"render failed: {type(exc).__name__}: {exc}")
        op_id = self._record(
            "render",
            {"path": str(resolved_path), "view": view, "shape_id": shape_id},
            [],
            f"Rendered {shape_id} to {resolved_path}",
        )
        return self._render_response(
            resolved_path,
            view,
            shape_id,
            int(width),
            int(height),
            operation_id=op_id,
        )

    def _render_response(
        self,
        path: Path,
        view: str,
        shape_id: Optional[str],
        width: int,
        height: int,
        operation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self._ok(
            summary=f"Rendered {shape_id or 'active shape'} to {path}",
            path=str(path),
            filename=path.name,
            size=path.stat().st_size if path.exists() else None,
            mime_type="image/png",
            view=view,
            width=width,
            height=height,
        )
        if operation_id is not None:
            result["operation_id"] = operation_id
        return result

    def add_parameter(
        self, name: str, value: float, units: str = "mm"
    ) -> Dict[str, Any]:
        """Compatibility wrapper for the persistent named-parameter API."""
        parameter_type = (
            "angle" if units.strip().lower() in {"deg", "rad"} else "length"
        )
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
        parameter_snapshots = [
            parameter.to_dict() for parameter in self.parameters.values()
        ]
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
            identifier = str(
                update.get("parameter_id") or update.get("name") or ""
            ).strip()
            parameter = self._find_parameter(identifier)
            if parameter is None:
                return self._error(f"Unknown parameter_id or name '{identifier}'.")
            if "value" not in update and "expression" not in update:
                return self._error(
                    f"Update for '{parameter.name}' requires value or expression."
                )
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
            item.to_dict()
            for item in self.parameters.values()
            if item.name in changed_names
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
            (
                item.to_dict()
                for item in self.parameters.values()
                if item.name == parameter_name
            ),
            None,
        )
        result.update(
            {
                "summary": (
                    f"Bound parameter '{parameter_name}' to {object_id}.{property_name}"
                ),
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

    def describe_state(self) -> Dict[str, Any]:
        if self._worker is not None:
            return self._worker.call("describe_state", {})
        return self._ok(
            summary=f"{len(self.objects)} objects, {len(self.operations)} operations",
            backend=self.backend_name,
            active_workplane_id=self.active_workplane_id,
            active_shape_id=self.active_shape_id,
            parameters=[parameter.to_dict() for parameter in self.parameters.values()],
            document=dict(self.document),
            document_revision=self.document_revision,
            objects=[obj.to_dict() for obj in self.objects.values()],
            operation_count=len(self.operations),
        )

    def list_objects(self) -> Dict[str, Any]:
        if self._worker is not None:
            return self._worker.call("list_objects", {})
        return self._ok(
            summary=f"{len(self.objects)} objects",
            objects=[obj.to_summary_dict() for obj in self.objects.values()],
            id_usage=(
                "Pass the complete id value, including its prefix, to get_object. "
                "For example: shape_1, not 1."
            ),
        )

    def get_object(self, object_id: str) -> Dict[str, Any]:
        """Return one semantic object by its RapidCAD ID."""
        if self._worker is not None:
            return self._worker.call("get_object", {"object_id": object_id})
        semantic_object = self.objects.get(object_id)
        if semantic_object is None:
            available_ids = list(self.objects)
            displayed_ids = available_ids[:20]
            available = ", ".join(repr(item) for item in displayed_ids)
            if len(available_ids) > len(displayed_ids):
                available += f", ... ({len(available_ids)} total)"
            guidance = (
                "Object IDs are exact and include their prefix. Copy the complete "
                "id from list_objects; do not derive an ID or use its numeric suffix."
            )
            if available:
                guidance += f" Available object_ids: {available}."
            else:
                guidance += " The active document currently has no objects."
            return self._error(f"Unknown object_id '{object_id}'. {guidance}")
        object_payload = semantic_object.to_dict()
        semantic_features = self._semantic_features_for_object(object_id)
        if semantic_features:
            object_payload["semantic_features"] = semantic_features
            object_payload["drawing_dimension_requests"] = [
                request
                for feature in semantic_features
                for request in self._drawing_dimension_requests(feature)
            ]
        return self._ok(
            summary=f"Object {object_id}",
            object=object_payload,
            document_revision=self.document_revision,
        )

    def _semantic_features_for_object(self, object_id: str) -> list[Dict[str, Any]]:
        """Expose authoritative feature IDs attached to an object's result chain."""

        if self.cad_document is None:
            return []
        runtime_object = self.runtime_objects.get(object_id)
        native = getattr(runtime_object, "native_handle", None)
        if native is None:
            return []
        native_names: set[str] = set()
        pending = [native]
        while pending:
            candidate = pending.pop()
            native_name = str(getattr(candidate, "Name", ""))
            if not native_name or native_name in native_names:
                continue
            native_names.add(native_name)
            base = getattr(candidate, "Base", None)
            if isinstance(base, tuple):
                pending.extend(item for item in base if item is not None)
            elif base is not None:
                pending.append(base)
        return [
            dict(value)
            for value in self.cad_document.feature_definitions_for_native_names(
                native_names
            )
        ]

    @staticmethod
    def _drawing_dimension_requests(
        feature: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        """Describe geometry-driven dimensions available for one feature."""

        if feature.get("kind") != "hole":
            return []
        axis = tuple(float(value) for value in feature.get("axis", (0, 0, 1)))
        absolute_axis = tuple(abs(value) for value in axis)
        largest_axis = absolute_axis.index(max(absolute_axis))
        recommended_view = ("right", "front", "top")[largest_axis]
        feature_id = str(feature.get("id", ""))
        requests = [
            {
                "feature_id": feature_id,
                "dimension_kind": "diameter",
                "geometry_reference": "hole_cylinder",
                "recommended_view": recommended_view,
                "measurement_source": "projected_geometry",
                "semantic_qualifiers": {
                    "termination": feature.get("termination"),
                    "depth_mm": feature.get("depth_mm"),
                },
            }
        ]
        if feature.get("hole_type") == "countersink":
            requests.append(
                {
                    "feature_id": feature_id,
                    "dimension_kind": "countersink_diameter",
                    "geometry_reference": "countersink_rim",
                    "recommended_view": recommended_view,
                    "measurement_source": "projected_geometry",
                    "semantic_qualifiers": {
                        "angle_degrees": feature.get(
                            "countersink_angle_degrees"
                        )
                    },
                }
            )
        return requests

    def set_object_property(
        self,
        object_id: str,
        property_name: str,
        value: Any,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Modify a property on the same live native CAD object."""
        if self._worker is not None:
            return self._worker.call(
                "set_object_property",
                {
                    "object_id": object_id,
                    "property_name": property_name,
                    "value": value,
                    "expected_revision": expected_revision,
                },
            )
        if expected_revision and expected_revision != self.document_revision:
            return self._error(
                "Document revision mismatch: expected "
                f"{expected_revision}, current {self.document_revision}."
            )
        runtime_object = self.runtime_objects.get(object_id)
        if not isinstance(runtime_object, CadObject):
            return self._error(f"Object '{object_id}' is not a live native CAD object.")
        try:
            runtime_object.set_property(property_name, value, recompute=True)
        except Exception as exc:
            return self._error(
                f"Could not set {object_id}.{property_name}: "
                f"{type(exc).__name__}: {exc}"
            )

        ids_by_name = {
            item.native_name: item_id
            for item_id, item in self.runtime_objects.items()
            if isinstance(item, CadObject)
        }
        serialized_value = self._serialize_freecad_value(
            runtime_object.get_property(property_name), ids_by_name
        )
        runtime_object.properties[property_name] = serialized_value
        shape = runtime_object.shape
        if shape is not None and not getattr(shape, "isNull", lambda: False)():
            geometry = self._shape_signature(shape)
            runtime_object.geometry = geometry
            self.geometry_signatures[object_id] = geometry
        semantic_object = self.objects[object_id]
        semantic_object.metadata["properties"] = runtime_object.properties
        if runtime_object.geometry:
            semantic_object.metadata["geometry"] = runtime_object.geometry
        self._recalculate_document_revision()
        op_id = self._record(
            "set_object_property",
            {
                "object_id": object_id,
                "property_name": property_name,
                "value": serialized_value,
                "expected_revision": expected_revision,
            },
            [object_id],
            f"Set {object_id}.{property_name}",
        )
        return self._ok(
            summary=f"Set {object_id}.{property_name}",
            operation_id=op_id,
            object_id=object_id,
            property_name=property_name,
            value=serialized_value,
            geometry=runtime_object.geometry,
            document_revision=self.document_revision,
            object=semantic_object.to_dict(),
        )

    def generate_drawing(
        self,
        object_ids: Optional[list[str]] = None,
        standard: str = "ISO",
        sheet_size: str = "A3",
        projection_angle: str = "first",
        template_id: Optional[str] = None,
        output_directory: Optional[str] = None,
        part_name: Optional[str] = None,
        run_id: Optional[str] = None,
        include_native: bool = True,
        dimension_feature_ids: Optional[list[str]] = None,
        output_formats: Optional[list[str]] = None,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a linked native drawing and export the requested formats."""

        ready = self._ensure_gui_session(require_document=True)
        if ready is not None:
            return ready
        params = {
            "object_ids": object_ids,
            "standard": standard,
            "sheet_size": sheet_size,
            "projection_angle": projection_angle,
            "template_id": template_id,
            "output_directory": output_directory,
            "part_name": part_name,
            "run_id": run_id,
            "include_native": include_native,
            "dimension_feature_ids": dimension_feature_ids,
            "output_formats": output_formats,
            "expected_revision": expected_revision,
        }
        if self._worker is not None:
            return self._worker.call("generate_drawing", params)
        if self.cad_document is None:
            return self._error("No native CAD document is open.")
        if expected_revision and expected_revision != self.document_revision:
            return self._error(
                "Document revision mismatch: expected "
                f"{expected_revision}, current {self.document_revision}."
            )

        requested_ids = list(object_ids or [])
        drawing_objects: list[CadObject] = []
        for object_id in requested_ids:
            runtime_object = self.runtime_objects.get(object_id)
            if not isinstance(runtime_object, CadObject):
                return self._error(
                    f"Object '{object_id}' is not a live native CAD object."
                )
            drawing_objects.append(runtime_object)

        resolved_output = (
            Path(output_directory).expanduser().resolve()
            if output_directory
            else Path(tempfile.mkdtemp(prefix="rapidcadpy_drawing_")).resolve()
        )
        resolved_part_name = (
            str(part_name).strip()
            if part_name
            else self.cad_document.label or self.cad_document.name or "drawing"
        )
        resolved_run_id = str(run_id).strip() if run_id else self._new_id("drawing")

        try:
            from .drawing import create_drawing_backend

            backend = create_drawing_backend(self.cad_document)
            drawing = backend.generate_drawing(
                document=self.cad_document,
                objects=drawing_objects,
                standard=standard,
                sheet_size=sheet_size,
                projection_angle=projection_angle,
                template_id=template_id,
                output_directory=resolved_output,
                part_name=resolved_part_name,
                run_id=resolved_run_id,
                include_native=include_native,
                dimension_feature_ids=dimension_feature_ids,
                output_formats=output_formats,
            )
        except Exception as exc:
            return self._error(
                f"Could not generate technical drawing: {type(exc).__name__}: {exc}"
            )

        native_document = self.cad_document.native_handle
        file_name = str(getattr(native_document, "FileName", "")).strip()
        file_path = Path(file_name).expanduser().resolve() if file_name else None
        hydrated = self._hydrate_freecad_document(
            native_document,
            file_path,
            source_tool="generate_drawing",
        )
        if not hydrated.get("ok"):
            return hydrated
        ids_by_native_name = {
            item.native_name: item_id
            for item_id, item in self.runtime_objects.items()
            if isinstance(item, CadObject)
        }
        created_object_ids = [
            ids_by_native_name[name]
            for name in drawing.created_native_names
            if name in ids_by_native_name
        ]
        views: list[Dict[str, Any]] = []
        items: list[Dict[str, Any]] = []
        view_lookup: Dict[str, str] = {}
        if drawing.page_name:
            try:
                inspection = backend.inspect_drawing(
                    document=self.cad_document,
                    page_name=str(drawing.page_name),
                )
                views = inspection["views"]
                items = inspection["items"]
                view_lookup = self._drawing_view_lookup(views)
            except Exception:
                pass
        view_choices = ", ".join(
            f"{name}={object_id}" for name, object_id in view_lookup.items()
        ) or "none"
        result = drawing.to_dict()
        result.update(
            {
                "ok": True,
                "summary": (
                    f"Created {standard.strip().upper()} "
                    f"{sheet_size.strip().upper()} {projection_angle.strip().lower()}-angle "
                    f"technical drawing for {resolved_part_name}. "
                    f"Available views: {view_choices}."
                ),
                "run_id": resolved_run_id,
                "created_object_ids": created_object_ids,
                "document_revision": self.document_revision,
                "views": views,
                "items": items,
                "view_lookup": view_lookup,
            }
        )
        self.drawings[resolved_run_id] = {
            **result,
            "drawing_id": resolved_run_id,
            "status": "partial" if result.get("warnings") else "complete",
            "workspace": str(resolved_output),
            "artifact_status": "current",
        }
        self.current_drawing_id = resolved_run_id
        return result

    def get_current_drawing(self) -> Dict[str, Any]:
        """Return the live session's latest drawing workspace manifest."""

        ready = self._ensure_gui_session(
            require_document=True,
            create_document_if_missing=False,
        )
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call("get_current_drawing", {})
        if self.current_drawing_id is None:
            discovered = self._discover_drawings()
            if isinstance(discovered, dict):
                return discovered
            if len(discovered) == 1:
                self.current_drawing_id = str(discovered[0]["drawing_id"])
            elif discovered:
                return self._error(
                    "Multiple drawing pages are open. Call list_drawings and then "
                    "select_drawing with the intended drawing_id."
                )
            else:
                return self._error("The active CAD document contains no drawing pages.")
        return self._ok(
            summary=f"Current drawing: {self.current_drawing_id}",
            drawing=dict(self.drawings[self.current_drawing_id]),
        )

    def list_drawings(self) -> Dict[str, Any]:
        """Discover native drawing pages in the currently opened CAD file."""

        ready = self._ensure_gui_session(
            require_document=True,
            create_document_if_missing=False,
        )
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call("list_drawings", {})
        discovered = self._discover_drawings()
        if isinstance(discovered, dict):
            return discovered
        return self._ok(
            summary=f"Found {len(discovered)} drawing page(s)",
            drawing_count=len(discovered),
            drawings=discovered,
            current_drawing_id=self.current_drawing_id,
            document_revision=self.document_revision,
        )

    def select_drawing(self, drawing_id: str) -> Dict[str, Any]:
        """Select one discovered drawing page for subsequent editing tools."""

        ready = self._ensure_gui_session(
            require_document=True,
            create_document_if_missing=False,
        )
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "select_drawing",
                {"drawing_id": drawing_id},
            )
        discovered = self._discover_drawings()
        if isinstance(discovered, dict):
            return discovered
        available = [str(item["drawing_id"]) for item in discovered]
        selected = self.drawings.get(str(drawing_id))
        if selected is None or str(drawing_id) not in available:
            return self._error(
                f"Unknown drawing_id {drawing_id!r}. Call list_drawings and copy "
                f"one exactly. Available drawing IDs: {available or 'none'}."
            )
        self.current_drawing_id = str(drawing_id)
        return self._ok(
            summary=f"Selected drawing {drawing_id}",
            drawing=dict(selected),
            document_revision=self.document_revision,
        )

    def list_drawing_items(
        self,
        drawing_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List exact view and editable-item IDs on an existing drawing."""

        ready = self._ensure_gui_session(
            require_document=True,
            create_document_if_missing=False,
        )
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "list_drawing_items",
                {"drawing_id": drawing_id},
            )
        resolved = self._resolve_drawing(drawing_id)
        if isinstance(resolved, dict) and resolved.get("ok") is False:
            return resolved
        drawing = resolved
        assert isinstance(drawing, dict)
        try:
            from .drawing import create_drawing_backend

            backend = create_drawing_backend(self.cad_document)
            inspection = backend.inspect_drawing(
                document=self.cad_document,
                page_name=str(drawing["page_name"]),
            )
        except Exception as exc:
            return self._error(
                f"Could not inspect drawing: {type(exc).__name__}: {exc}"
            )
        drawing["views"] = inspection["views"]
        drawing["items"] = inspection["items"]
        view_lookup = self._drawing_view_lookup(inspection["views"])
        view_choices = ", ".join(
            f"{name}={object_id}" for name, object_id in view_lookup.items()
        )
        if not view_choices:
            view_choices = ", ".join(
                f"{view.get('native_name', 'view')}={view.get('id')}"
                for view in inspection["views"]
            ) or "none"
        return self._ok(
            summary=(
                f"Drawing {drawing['drawing_id']} has "
                f"{len(inspection['views'])} views and "
                f"{len(inspection['items'])} editable items. "
                f"Available views: {view_choices}."
            ),
            drawing_id=drawing["drawing_id"],
            page_id=inspection["page_id"],
            views=inspection["views"],
            view_lookup=view_lookup,
            items=inspection["items"],
            document_revision=self.document_revision,
        )

    def add_feature_dimension(
        self,
        feature_id: str,
        dimension_kind: str,
        view_id: str,
        position_mm: list[float],
        drawing_id: Optional[str] = None,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a native dimension whose numeric value comes from geometry."""

        params = {
            "feature_id": feature_id,
            "dimension_kind": dimension_kind,
            "view_id": view_id,
            "position_mm": position_mm,
            "drawing_id": drawing_id,
            "expected_revision": expected_revision,
        }
        if self._worker is not None:
            return self._worker.call("add_feature_dimension", params)
        ready = self._prepare_drawing_edit(drawing_id, expected_revision)
        if isinstance(ready, dict) and ready.get("ok") is False:
            return ready
        drawing = ready
        assert isinstance(drawing, dict)
        definition = next(
            (
                dict(value)
                for value in self.cad_document.feature_definitions.values()
                if str(value.get("id", "")) == str(feature_id)
            ),
            None,
        )
        if definition is None:
            available = sorted(
                str(value.get("id"))
                for value in self.cad_document.feature_definitions.values()
                if value.get("id")
            )
            return self._error(
                f"Unknown feature_id {feature_id!r}. Call get_object and copy "
                f"semantic_features[].id exactly. Available IDs: {available or 'none'}."
            )
        view = self._resolve_drawing_view(drawing, view_name=None, view_id=view_id)
        if isinstance(view, dict):
            return view
        try:
            from .drawing import create_drawing_backend

            backend = create_drawing_backend(self.cad_document)
            edit = backend.add_feature_dimension(
                document=self.cad_document,
                page_name=str(drawing["page_name"]),
                view_native_name=view.native_name,
                feature_definition=definition,
                dimension_kind=dimension_kind,
                position_mm=self._drawing_position(position_mm),
                standard=str(
                    (drawing.get("metadata") or {}).get("standard", "ISO")
                ),
            )
        except Exception as exc:
            return self._error(
                f"Could not add feature dimension: {type(exc).__name__}: {exc}"
            )
        return self._complete_drawing_edit(
            drawing,
            "add_feature_dimension",
            params,
            edit,
            f"Added geometry-measured {dimension_kind} dimension",
        )

    def add_drawing_note(
        self,
        text: str,
        position_mm: list[float],
        drawing_id: Optional[str] = None,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add an editorial native TechDraw annotation."""

        params = {
            "text": text,
            "position_mm": position_mm,
            "drawing_id": drawing_id,
            "expected_revision": expected_revision,
        }
        if self._worker is not None:
            return self._worker.call("add_drawing_note", params)
        ready = self._prepare_drawing_edit(drawing_id, expected_revision)
        if isinstance(ready, dict) and ready.get("ok") is False:
            return ready
        drawing = ready
        assert isinstance(drawing, dict)
        try:
            from .drawing import create_drawing_backend

            edit = create_drawing_backend(self.cad_document).add_drawing_note(
                document=self.cad_document,
                page_name=str(drawing["page_name"]),
                text=text,
                position_mm=self._drawing_position(position_mm),
            )
        except Exception as exc:
            return self._error(f"Could not add drawing note: {type(exc).__name__}: {exc}")
        return self._complete_drawing_edit(
            drawing,
            "add_drawing_note",
            params,
            edit,
            "Added native drawing note",
        )

    def add_drawing_leader(
        self,
        text: str,
        view_name: Optional[str] = None,
        view_id: Optional[str] = None,
        anchor_mm: Optional[list[float]] = None,
        elbow_mm: Optional[list[float]] = None,
        text_position_mm: Optional[list[float]] = None,
        drawing_id: Optional[str] = None,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a native leader and label to a projected drawing view."""

        params = {
            "text": text,
            "view_name": view_name,
            "view_id": view_id,
            "anchor_mm": anchor_mm,
            "elbow_mm": elbow_mm,
            "text_position_mm": text_position_mm,
            "drawing_id": drawing_id,
            "expected_revision": expected_revision,
        }
        if self._worker is not None:
            return self._worker.call("add_drawing_leader", params)
        ready = self._prepare_drawing_edit(drawing_id, expected_revision)
        if isinstance(ready, dict) and ready.get("ok") is False:
            return ready
        drawing = ready
        assert isinstance(drawing, dict)
        view = self._resolve_drawing_view(
            drawing,
            view_name=view_name,
            view_id=view_id,
        )
        if isinstance(view, dict):
            return view
        try:
            from .drawing import create_drawing_backend

            edit = create_drawing_backend(self.cad_document).add_drawing_leader(
                document=self.cad_document,
                page_name=str(drawing["page_name"]),
                view_native_name=view.native_name,
                text=text,
                anchor_mm=(
                    self._drawing_position(anchor_mm) if anchor_mm is not None else None
                ),
                elbow_mm=(
                    self._drawing_position(elbow_mm) if elbow_mm is not None else None
                ),
                text_position_mm=(
                    self._drawing_position(text_position_mm)
                    if text_position_mm is not None
                    else None
                ),
            )
        except Exception as exc:
            return self._error(
                f"Could not add drawing leader: {type(exc).__name__}: {exc}"
            )
        return self._complete_drawing_edit(
            drawing,
            "add_drawing_leader",
            params,
            edit,
            "Added native drawing leader and label",
        )

    def move_drawing_item(
        self,
        item_id: str,
        position_mm: list[float],
        drawing_id: Optional[str] = None,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Move an existing RapidCAD dimension, note, or leader."""

        params = {
            "item_id": item_id,
            "position_mm": position_mm,
            "drawing_id": drawing_id,
            "expected_revision": expected_revision,
        }
        if self._worker is not None:
            return self._worker.call("move_drawing_item", params)
        ready = self._prepare_drawing_edit(drawing_id, expected_revision)
        if isinstance(ready, dict) and ready.get("ok") is False:
            return ready
        drawing = ready
        assert isinstance(drawing, dict)
        item = self._drawing_runtime_object(item_id, "item")
        if isinstance(item, dict):
            return item
        try:
            from .drawing import create_drawing_backend

            edit = create_drawing_backend(self.cad_document).move_drawing_item(
                document=self.cad_document,
                page_name=str(drawing["page_name"]),
                item_native_name=item.native_name,
                position_mm=self._drawing_position(position_mm),
            )
        except Exception as exc:
            return self._error(
                f"Could not move drawing item: {type(exc).__name__}: {exc}"
            )
        return self._complete_drawing_edit(
            drawing,
            "move_drawing_item",
            params,
            edit,
            f"Moved drawing item {item_id}",
        )

    def export_current_drawing(
        self,
        drawing_id: Optional[str] = None,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Re-export the edited page to its managed PDF and SVG paths."""

        params = {
            "drawing_id": drawing_id,
            "expected_revision": expected_revision,
        }
        if self._worker is not None:
            return self._worker.call("export_current_drawing", params)
        ready = self._prepare_drawing_edit(drawing_id, expected_revision)
        if isinstance(ready, dict) and ready.get("ok") is False:
            return ready
        drawing = ready
        assert isinstance(drawing, dict)
        pdf_path = Path(str(drawing.get("pdf_path") or "")).expanduser()
        if not str(drawing.get("pdf_path") or "").strip():
            return self._error("The current drawing has no managed PDF path.")
        raw_vector = str(drawing.get("vector_source_path") or "").strip()
        vector_path = (
            Path(raw_vector).expanduser()
            if raw_vector
            else Path(str(drawing["workspace"])) / ".intermediate" / ".edited.svg"
        )
        try:
            from .drawing import create_drawing_backend

            edit = create_drawing_backend(self.cad_document).export_drawing(
                document=self.cad_document,
                page_name=str(drawing["page_name"]),
                pdf_path=pdf_path,
                vector_source_path=vector_path,
            )
        except Exception as exc:
            return self._error(
                f"Could not export current drawing: {type(exc).__name__}: {exc}"
            )
        drawing["artifact_status"] = "current"
        drawing["vector_source_path"] = str(vector_path)
        result = edit.to_dict()
        result.update(
            {
                "ok": True,
                "summary": f"Exported drawing {drawing['drawing_id']}",
                "drawing_id": drawing["drawing_id"],
                "pdf_path": str(pdf_path),
                "vector_source_path": str(vector_path),
                "document_revision": self.document_revision,
            }
        )
        return result

    def _resolve_drawing(
        self,
        drawing_id: Optional[str],
    ) -> Dict[str, Any]:
        if self.cad_document is None:
            return self._error("No native CAD document is open.")
        discovered = self._discover_drawings()
        if isinstance(discovered, dict):
            return discovered
        resolved_id = str(drawing_id).strip() if drawing_id else self.current_drawing_id
        if not resolved_id and len(discovered) == 1:
            resolved_id = str(discovered[0]["drawing_id"])
            self.current_drawing_id = resolved_id
        if not resolved_id and len(discovered) > 1:
            return self._error(
                "Multiple drawing pages are open. Call list_drawings and then "
                "select_drawing before editing."
            )
        discovered_ids = {str(item["drawing_id"]) for item in discovered}
        if (
            not resolved_id
            or resolved_id not in self.drawings
            or resolved_id not in discovered_ids
        ):
            return self._error(
                "No matching drawing exists in the active CAD document. Call "
                "list_drawings to inspect existing pages or generate_drawing to "
                "create one."
            )
        drawing = self.drawings[resolved_id]
        if not drawing.get("page_name"):
            return self._error(f"Drawing {resolved_id!r} has no native page reference.")
        return drawing

    def _discover_drawings(self) -> list[Dict[str, Any]] | Dict[str, Any]:
        if self.cad_document is None:
            return self._error("No native CAD document is open.")
        native_document = self.cad_document.native_handle
        file_name = str(getattr(native_document, "FileName", "")).strip()
        file_path = Path(file_name).expanduser().resolve() if file_name else None
        hydrated = self._hydrate_freecad_document(
            native_document,
            file_path,
            source_tool="drawing_discovery",
        )
        if not hydrated.get("ok"):
            return hydrated
        try:
            from .drawing import create_drawing_backend

            discovered = create_drawing_backend(self.cad_document).list_drawings(
                document=self.cad_document
            )
        except Exception as exc:
            return self._error(
                f"Could not discover drawing pages: {type(exc).__name__}: {exc}"
            )
        for item in discovered:
            drawing_id = str(item["drawing_id"])
            existing = self.drawings.get(drawing_id, {})
            self.drawings[drawing_id] = {
                **item,
                **existing,
                "drawing_id": drawing_id,
                "page_id": item["page_id"],
                "page_name": item["page_name"],
                "label": item["label"],
                "managed_by_rapidcad": item["managed_by_rapidcad"],
                "views": item["views"],
                "items": item["items"],
                "status": existing.get("status", "opened"),
                "artifact_status": existing.get("artifact_status", "unmanaged"),
            }
        return [dict(self.drawings[str(item["drawing_id"])]) for item in discovered]

    def _prepare_drawing_edit(
        self,
        drawing_id: Optional[str],
        expected_revision: Optional[str],
    ) -> Dict[str, Any]:
        drawing = self._resolve_drawing(drawing_id)
        if drawing.get("ok") is False:
            return drawing
        if expected_revision and expected_revision != self.document_revision:
            return self._error(
                "Document revision mismatch: expected "
                f"{expected_revision}, current {self.document_revision}."
            )
        return drawing

    def _drawing_runtime_object(
        self,
        object_id: str,
        expected_kind: str,
    ) -> CadObject | Dict[str, Any]:
        runtime_object = self.runtime_objects.get(object_id)
        if not isinstance(runtime_object, CadObject):
            return self._error(
                f"Unknown drawing {expected_kind}_id {object_id!r}. Call "
                "list_drawing_items and copy the complete id exactly."
            )
        return runtime_object

    @staticmethod
    def _drawing_view_lookup(views: list[Dict[str, Any]]) -> Dict[str, str]:
        """Return unambiguous semantic drawing-view names mapped to exact IDs."""

        grouped: Dict[str, list[str]] = {}
        for view in views:
            name = str(view.get("view_name") or "").strip().lower()
            object_id = str(view.get("id") or "").strip()
            if name and object_id:
                grouped.setdefault(name, []).append(object_id)
        return {
            name: object_ids[0]
            for name, object_ids in grouped.items()
            if len(object_ids) == 1
        }

    def _resolve_drawing_view(
        self,
        drawing: Dict[str, Any],
        *,
        view_name: Optional[str],
        view_id: Optional[str],
    ) -> CadObject | Dict[str, Any]:
        """Resolve a semantic view name or an exact runtime view ID."""

        views = list(drawing.get("views") or [])
        lookup = self._drawing_view_lookup(views)
        normalized_name = str(view_name or "").strip().lower()
        normalized_id = str(view_id or "").strip()
        if not normalized_name and not normalized_id:
            return self._error(
                "A drawing view is required. Pass view_name from "
                f"list_drawing_items.view_lookup. Available views: {lookup or 'none'}."
            )
        if normalized_name:
            matching_ids = [
                str(view.get("id"))
                for view in views
                if str(view.get("view_name") or "").strip().lower()
                == normalized_name
                and view.get("id")
            ]
            if not matching_ids:
                return self._error(
                    f"Unknown drawing view_name {view_name!r}. Available semantic "
                    f"views: {lookup or 'none'}."
                )
            if len(matching_ids) > 1 and not normalized_id:
                return self._error(
                    f"Drawing view_name {view_name!r} is ambiguous. Pass one exact "
                    f"view_id from {matching_ids}."
                )
            semantic_id = matching_ids[0] if len(matching_ids) == 1 else normalized_id
            if normalized_id and normalized_id not in matching_ids:
                return self._error(
                    f"view_id {view_id!r} does not identify the {normalized_name!r} "
                    f"view. Matching IDs: {matching_ids}."
                )
            normalized_id = semantic_id
        runtime_object = self.runtime_objects.get(normalized_id)
        if not isinstance(runtime_object, CadObject) and not normalized_name:
            # A caller sometimes passes a guessed semantic name (for example
            # "FrontView") in the view_id slot instead of view_name. Recover
            # it against the lookup before failing outright.
            guessed_name = normalized_id.lower()
            if guessed_name.endswith("view"):
                guessed_name = guessed_name[: -len("view")].strip()
            guessed_id = lookup.get(guessed_name)
            if guessed_id:
                normalized_id = guessed_id
                runtime_object = self.runtime_objects.get(normalized_id)
        if not isinstance(runtime_object, CadObject):
            choices = {
                str(view.get("view_name") or view.get("native_name") or "view"): str(
                    view.get("id")
                )
                for view in views
                if view.get("id")
            }
            return self._error(
                f"Unknown drawing view_id {view_id!r}. Available views: "
                f"{choices or 'none'}. Prefer view_name when it is available."
            )
        return runtime_object

    @staticmethod
    def _drawing_position(value: list[float]) -> tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("Drawing positions must be [x_mm, y_mm].")
        return float(value[0]), float(value[1])

    def _complete_drawing_edit(
        self,
        drawing: Dict[str, Any],
        operation: str,
        params: Dict[str, Any],
        edit: Any,
        summary: str,
    ) -> Dict[str, Any]:
        native_document = self.cad_document.native_handle
        file_name = str(getattr(native_document, "FileName", "")).strip()
        file_path = Path(file_name).expanduser().resolve() if file_name else None
        hydrated = self._hydrate_freecad_document(
            native_document,
            file_path,
            source_tool=operation,
        )
        if not hydrated.get("ok"):
            return hydrated
        ids_by_name = {
            item.native_name: object_id
            for object_id, item in self.runtime_objects.items()
            if isinstance(item, CadObject)
        }
        created_ids = [
            ids_by_name[name]
            for name in edit.created_native_names
            if name in ids_by_name
        ]
        changed_ids = [
            ids_by_name[name]
            for name in edit.changed_native_names
            if name in ids_by_name
        ]
        drawing["artifact_status"] = "stale"
        drawing["status"] = "edited"
        operation_id = self._record(
            operation,
            params,
            created_ids or changed_ids,
            summary,
        )
        result = edit.to_dict()
        result.update(
            {
                "ok": True,
                "summary": summary,
                "operation_id": operation_id,
                "drawing_id": drawing["drawing_id"],
                "created_object_ids": created_ids,
                "changed_object_ids": changed_ids,
                "document_revision": self.document_revision,
                "artifact_status": "stale",
                "next_action": (
                    "Call export_current_drawing to refresh the downloadable PDF."
                ),
            }
        )
        return result

    def save_document(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Save the active native document, optionally to a new path."""
        if self._worker is not None:
            return self._worker.call("save_document", {"path": path})
        if self.cad_document is None:
            return self._error("No native CAD document is open.")
        if not path and not str(getattr(self.cad_document.native_handle, "FileName", "")).strip():
            return self._error(
                "This document has not been saved before. Provide a path to save_document."
            )
        try:
            saved_path = self.cad_document.save(path)
        except Exception as exc:
            return self._error(f"Could not save document: {type(exc).__name__}: {exc}")
        self.document["file_name"] = saved_path
        op_id = self._record(
            "save_document",
            {"path": saved_path},
            [],
            f"Saved FreeCAD document to {saved_path}",
        )
        return self._ok(
            summary=f"Saved FreeCAD document to {saved_path}",
            operation_id=op_id,
            path=saved_path,
            document_revision=self.document_revision,
        )

    def select_object(self, object_id: str) -> Dict[str, Any]:
        """Select a native object in a connected FreeCAD GUI."""
        if self._gui_connection is None:
            return self._error("Object selection requires FreeCAD GUI mode.")
        return self._gui_connection.call("select_object", {"object_id": object_id})

    def fit_view(self) -> Dict[str, Any]:
        """Fit the connected FreeCAD GUI view to the active model."""
        if self._gui_connection is None:
            return self._error("fit_view requires FreeCAD GUI mode.")
        return self._gui_connection.call("fit_view", {})

    def get_history(self, limit: int = 50) -> Dict[str, Any]:
        if self._worker is not None:
            return self._worker.call("get_history", {"limit": limit})
        records = self.operations[-int(limit) :]
        return self._ok(
            summary=f"{len(records)} operations returned",
            operations=[op.to_dict() for op in records],
        )

    def _export(
        self, path: str, kind: str, shape_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if self._worker is not None:
            return self._worker.call(
                f"export_{kind}", {"path": path, "shape_id": shape_id}
            )
        ready = self._require_app()
        if ready is not None:
            return ready
        shape_id = shape_id or self.active_shape_id
        resolved_path = self._resolve_export_path(path)
        try:
            self._ensure_parent(str(resolved_path))
            if shape_id:
                shape = self._shape_wrappers.get(shape_id)
                if shape is None:
                    return self._error(
                        f"Object '{shape_id}' has no geometry export wrapper."
                    )
                if kind == "step":
                    shape.to_step(str(resolved_path))
                elif kind == "stl":
                    shape.to_stl(str(resolved_path))
                else:
                    shape.to_fcstd(str(resolved_path))
            else:
                if kind == "step":
                    self.app.to_step(str(resolved_path))
                elif kind == "stl":
                    self.app.to_stl(str(resolved_path))
                else:
                    self.app.to_fcstd(str(resolved_path))
        except Exception as exc:
            return self._error(f"export_{kind} failed: {type(exc).__name__}: {exc}")
        op_id = self._record(
            f"export_{kind}",
            {"path": str(resolved_path), "shape_id": shape_id},
            [],
            f"Exported {kind} to {resolved_path}",
        )
        return self._ok(
            summary=f"Exported {kind} to {resolved_path}",
            operation_id=op_id,
            path=str(resolved_path),
            filename=resolved_path.name,
            size=resolved_path.stat().st_size if resolved_path.exists() else None,
        )

    def _geometry_signature(self, shape: Any) -> Dict[str, Any]:
        obj = getattr(shape, "obj", None)
        if obj is None and isinstance(shape, CadObject):
            obj = shape.shape
        if obj is None:
            return {}
        return self._shape_signature(obj)

    def _hydrate_freecad_document(
        self,
        doc: Any,
        file_path: Optional[Path],
        source_tool: str = "open_document",
    ) -> Dict[str, Any]:
        """Build the semantic mirror while retaining native DocumentObjects."""
        self.active_workplane = None
        self.active_workplane_id = None
        self.active_shape_id = None
        self.objects.clear()
        self.runtime_objects.clear()
        self.operations.clear()
        self.parameters.clear()
        self.geometry_signatures.clear()

        native_objects = list(getattr(doc, "Objects", []))
        cad_document = self.cad_document
        if cad_document is None or cad_document.native_handle is not doc:
            cad_document = getattr(self.app, "cad_document", None)
        if cad_document is None or cad_document.native_handle is not doc:
            from rapidcadpy.integrations.freecad.cad_adapter import FreeCADAdapter

            adapter = FreeCADAdapter()
            cad_document = CadDocument(
                backend=adapter.backend_name,
                native_handle=doc,
                adapter=adapter,
                name=str(getattr(doc, "Name", "")),
                label=str(getattr(doc, "Label", "")),
                file_name=str(getattr(doc, "FileName", "") or file_path or ""),
            )
        self.cad_document = cad_document
        parameter_adapter = getattr(cad_document.adapter, "parameter_adapter", None)

        ids_by_name: Dict[str, str] = {}
        for native_obj in native_objects:
            native_name = str(native_obj.Name)
            ids_by_name[native_name] = cad_document.object_id(
                native_name,
                lambda: self._new_id("native"),
            )

        containment = self._freecad_containment(native_objects, ids_by_name)
        rapidcad_ids = list(ids_by_name.values())
        source_path = str(file_path) if file_path is not None else ""
        op_id = self._record(
            source_tool,
            {"path": source_path} if source_path else {},
            rapidcad_ids,
            f"Hydrated {len(native_objects)} native FreeCAD objects",
        )

        for native_obj in native_objects:
            rapidcad_id = ids_by_name[str(native_obj.Name)]
            semantic_type = self._freecad_semantic_type(native_obj)
            geometry: Dict[str, Any] = {}
            shape = getattr(native_obj, "Shape", None)
            if shape is not None and not getattr(shape, "isNull", lambda: False)():
                geometry = self._shape_signature(shape)
                self.geometry_signatures[rapidcad_id] = geometry

            depends_on = self._freecad_object_ids(
                getattr(native_obj, "OutList", []), ids_by_name
            )
            dependents = self._freecad_object_ids(
                getattr(native_obj, "InList", []), ids_by_name
            )
            properties = self._freecad_properties(native_obj, ids_by_name)
            capabilities = {"get_properties", "set_properties"}
            if geometry:
                capabilities.add("geometry")
            if semantic_type in {"feature", "sketch"}:
                if str(getattr(native_obj, "TypeId", "")) not in {
                    "Part::Feature",
                    "PartDesign::Feature",
                }:
                    capabilities.add("feature_history")
                wrapper_type = CadFeature
            else:
                wrapper_type = CadObject
            if parameter_adapter is not None:
                for binding_name in parameter_adapter.supported_feature_properties(
                    native_obj
                ):
                    capabilities.add(f"parameter.bind:{binding_name}")
            cad_object = wrapper_type(
                id=rapidcad_id,
                document=cad_document,
                native_handle=native_obj,
                native_name=str(getattr(native_obj, "Name", "")),
                native_type=str(getattr(native_obj, "TypeId", "")),
                label=str(getattr(native_obj, "Label", native_obj.Name)),
                semantic_type=semantic_type,
                visibility=getattr(
                    getattr(native_obj, "ViewObject", None), "Visibility", None
                ),
                properties=properties,
                depends_on=depends_on,
                dependents=dependents,
                parent_ids=containment["parents"].get(rapidcad_id, []),
                child_ids=containment["children"].get(rapidcad_id, []),
                geometry=geometry,
                capabilities=frozenset(capabilities),
            )
            self.runtime_objects[rapidcad_id] = cad_object
            metadata: Dict[str, Any] = {
                "native_name": cad_object.native_name,
                "native_type": cad_object.native_type,
                "backend": cad_object.backend,
                "visibility": cad_object.visibility,
                "properties": cad_object.properties,
                "dependencies": {
                    "depends_on": cad_object.depends_on,
                    "dependents": cad_object.dependents,
                },
                "parent_ids": cad_object.parent_ids,
                "child_ids": cad_object.child_ids,
                "capabilities": sorted(cad_object.capabilities),
            }
            if geometry:
                metadata["geometry"] = geometry
            self.objects[rapidcad_id] = SemanticObject(
                id=rapidcad_id,
                type=semantic_type,
                label=cad_object.label,
                source_op=op_id,
                source="native_document",
                metadata=metadata,
            )

        cad_document.feature_definitions.clear()
        for native_obj in native_objects:
            raw_definition = getattr(native_obj, "RapidCADFeatureDefinition", "")
            if not raw_definition:
                continue
            try:
                definition = json.loads(str(raw_definition))
                if isinstance(definition, dict):
                    cad_document.register_feature_definition(native_obj.Name, definition)
            except (TypeError, ValueError, json.JSONDecodeError):
                # Invalid third-party metadata must not prevent document hydration.
                continue

        object_snapshots = [obj.to_dict() for obj in self.objects.values()]
        if parameter_adapter is not None:
            discovered_parameters = parameter_adapter.discover_parameters(
                doc,
                cad_document,
                ids_by_name,
                lambda: self._new_id("parameter"),
            )
            self.parameters = {
                parameter.id: parameter for parameter in discovered_parameters
            }
        parameter_snapshots = [
            parameter.to_dict() for parameter in self.parameters.values()
        ]
        revision_payload = json.dumps(
            {
                "objects": object_snapshots,
                "parameters": parameter_snapshots,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.document_revision = (
            "sha256:" + hashlib.sha256(revision_payload).hexdigest()
        )
        cad_document.revision = self.document_revision
        cad_document.name = str(getattr(doc, "Name", ""))
        cad_document.label = str(getattr(doc, "Label", ""))
        cad_document.file_name = str(getattr(doc, "FileName", "") or source_path)
        self.document = {
            key: value
            for key, value in cad_document.to_dict().items()
            if key != "backend"
        }
        tree = self._freecad_tree(
            rapidcad_ids, containment["children"], containment["parents"]
        )
        return self._ok(
            summary=(
                f"Hydrated active FreeCAD document with {len(native_objects)} objects"
                if source_tool == "use_active_document"
                else (
                    (
                        "Opened and hydrated FreeCAD document with "
                        f"{len(native_objects)} objects"
                    )
                    if source_tool == "open_document"
                    else (
                        f"Hydrated FreeCAD document after {source_tool} with "
                        f"{len(native_objects)} objects"
                    )
                )
            ),
            operation_id=op_id,
            document=dict(self.document),
            document_revision=self.document_revision,
            object_count=len(native_objects),
            objects=object_snapshots,
            parameter_count=len(parameter_snapshots),
            parameters=parameter_snapshots,
            tree=tree,
        )

    def _freecad_properties(
        self, obj: Any, ids_by_name: Dict[str, str]
    ) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        for name in getattr(obj, "PropertiesList", []):
            try:
                value = obj.getPropertyByName(name)
                if name == "Shape":
                    signature = self._shape_signature(value)
                    properties[name] = {
                        "kind": "geometry",
                        "signature": signature,
                    }
                else:
                    properties[name] = self._serialize_freecad_value(value, ids_by_name)
            except Exception as exc:
                properties[name] = {"unavailable": f"{type(exc).__name__}: {exc}"}
        return properties

    def _serialize_freecad_value(
        self, value: Any, ids_by_name: Dict[str, str], depth: int = 0
    ) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if depth >= 4:
            return str(value)
        native_name = getattr(value, "Name", None)
        if native_name is not None and str(native_name) in ids_by_name:
            return {"object_id": ids_by_name[str(native_name)]}
        if isinstance(value, (list, tuple)):
            return [
                self._serialize_freecad_value(item, ids_by_name, depth + 1)
                for item in value
            ]
        if isinstance(value, dict):
            return {
                str(key): self._serialize_freecad_value(item, ids_by_name, depth + 1)
                for key, item in value.items()
            }
        if all(hasattr(value, attr) for attr in ("x", "y", "z")):
            return {
                "x": float(value.x),
                "y": float(value.y),
                "z": float(value.z),
            }
        if hasattr(value, "Base") and hasattr(value, "Rotation"):
            placement: Dict[str, Any] = {
                "base": self._serialize_freecad_value(
                    value.Base, ids_by_name, depth + 1
                )
            }
            try:
                placement["rotation_quaternion"] = [
                    float(component) for component in value.Rotation.Q
                ]
            except Exception:
                placement["rotation"] = str(value.Rotation)
            return placement
        if hasattr(value, "Value"):
            quantity: Dict[str, Any] = {"value": float(value.Value)}
            unit = str(getattr(value, "Unit", ""))
            if unit:
                quantity["unit"] = unit
            return quantity
        return str(value)

    def _freecad_object_ids(
        self, native_objects: Any, ids_by_name: Dict[str, str]
    ) -> list[str]:
        result = []
        for native_obj in native_objects or []:
            rapidcad_id = ids_by_name.get(str(getattr(native_obj, "Name", "")))
            if rapidcad_id is not None:
                result.append(rapidcad_id)
        return result

    def _freecad_containment(
        self, native_objects: list[Any], ids_by_name: Dict[str, str]
    ) -> Dict[str, Dict[str, list[str]]]:
        children: Dict[str, list[str]] = {item: [] for item in ids_by_name.values()}
        parents: Dict[str, list[str]] = {item: [] for item in ids_by_name.values()}
        for native_obj in native_objects:
            parent_id = ids_by_name[str(native_obj.Name)]
            members = getattr(native_obj, "Group", []) or []
            for child_id in self._freecad_object_ids(members, ids_by_name):
                if child_id not in children[parent_id]:
                    children[parent_id].append(child_id)
                if parent_id not in parents[child_id]:
                    parents[child_id].append(parent_id)

            try:
                geo_parent = native_obj.getParentGeoFeatureGroup()
            except Exception:
                geo_parent = None
            if geo_parent is not None:
                geo_parent_id = ids_by_name.get(str(getattr(geo_parent, "Name", "")))
                if geo_parent_id and parent_id not in children[geo_parent_id]:
                    children[geo_parent_id].append(parent_id)
                    parents[parent_id].append(geo_parent_id)
        return {"children": children, "parents": parents}

    def _freecad_tree(
        self,
        object_ids: list[str],
        children: Dict[str, list[str]],
        parents: Dict[str, list[str]],
    ) -> list[Dict[str, Any]]:
        def node(object_id: str, ancestors: set[str]) -> Dict[str, Any]:
            if object_id in ancestors:
                return {"id": object_id, "cycle": True, "children": []}
            next_ancestors = set(ancestors)
            next_ancestors.add(object_id)
            return {
                "id": object_id,
                "children": [
                    node(item, next_ancestors) for item in children[object_id]
                ],
            }

        roots = [object_id for object_id in object_ids if not parents[object_id]]
        return [node(object_id, set()) for object_id in roots]

    def _freecad_semantic_type(self, obj: Any) -> str:
        type_id = str(getattr(obj, "TypeId", ""))
        if type_id == "App::Part":
            return "part"
        if type_id == "PartDesign::Body":
            return "body"
        if "Sketch" in type_id:
            return "sketch"
        if type_id.startswith(("Part::", "PartDesign::")):
            return "feature"
        return "native_object"

    def _find_parameter(self, identifier: str) -> Optional[CadParameter]:
        if identifier in self.parameters:
            return self.parameters[identifier]
        return next(
            (
                parameter
                for parameter in self.parameters.values()
                if parameter.name == identifier
            ),
            None,
        )

    def _parameter_context(self) -> tuple[Any, Any]:
        if self.cad_document is None:
            raise RuntimeError("No native CAD document is open.")
        adapter = getattr(self.cad_document.adapter, "parameter_adapter", None)
        if adapter is None:
            raise NotImplementedError(
                f"{self.cad_document.backend} does not support persistent "
                "named parameters."
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
        object_snapshots = [obj.to_dict() for obj in self.objects.values()]
        parameter_snapshots = [
            parameter.to_dict() for parameter in self.parameters.values()
        ]
        revision_payload = json.dumps(
            {
                "objects": object_snapshots,
                "parameters": parameter_snapshots,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.document_revision = (
            "sha256:" + hashlib.sha256(revision_payload).hexdigest()
        )
        if self.cad_document is not None:
            self.cad_document.revision = self.document_revision
        self.document["revision"] = self.document_revision
        return self.document_revision

    def _describe_freecad_object(self, obj: Any) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "name": getattr(obj, "Name", None),
            "label": getattr(obj, "Label", None),
            "type_id": getattr(obj, "TypeId", None),
            "visibility": getattr(getattr(obj, "ViewObject", None), "Visibility", None),
        }
        shape = getattr(obj, "Shape", None)
        if shape is not None and not getattr(shape, "isNull", lambda: False)():
            item["geometry"] = self._shape_signature(shape)
        return item

    def _shape_signature(self, obj: Any) -> Dict[str, Any]:
        signature: Dict[str, Any] = {}
        try:
            bb = obj.BoundBox
            signature["bbox"] = {
                "x_min": float(bb.XMin),
                "x_max": float(bb.XMax),
                "y_min": float(bb.YMin),
                "y_max": float(bb.YMax),
                "z_min": float(bb.ZMin),
                "z_max": float(bb.ZMax),
                "x_length": float(bb.XLength),
                "y_length": float(bb.YLength),
                "z_length": float(bb.ZLength),
            }
        except Exception:
            pass
        for attr, key in (("Volume", "volume"), ("Area", "area")):
            try:
                signature[key] = float(getattr(obj, attr))
            except Exception:
                pass
        for attr, key in (
            ("Faces", "face_count"),
            ("Edges", "edge_count"),
            ("Vertexes", "vertex_count"),
        ):
            try:
                signature[key] = len(getattr(obj, attr))
            except Exception:
                pass
        return signature

    def _record(
        self, tool: str, args: Dict[str, Any], outputs: list[str], summary: str
    ) -> str:
        op_id = self._new_id("op")
        self.operations.append(
            OperationRecord(
                id=op_id, tool=tool, args=args, outputs=outputs, summary=summary
            )
        )
        return op_id

    def _new_id(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}_{self._counters[prefix]}"

    def _ensure_gui_session(
        self,
        *,
        require_document: bool = True,
        create_document_if_missing: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Attach GUI-mode sessions automatically when one instance is available."""
        if (
            self.execution_mode != "gui"
            or self._worker is not None
            or self.app is not None
        ):
            return None

        attached = self.attach_freecad(use_active_document=require_document)
        if not attached.get("ok"):
            return attached
        if self._worker is None:
            return self._error(
                "FreeCAD attachment did not create a live GUI connection."
            )
        if not require_document or not attached.get("warning"):
            return None
        if attached.get("active_document") is None:
            if not create_document_if_missing:
                return self._error(
                    "FreeCAD has no active document. Open the intended CAD file "
                    "or call open_document before drawing discovery."
                )
            created = self._worker.call("new_document", {"name": "RapidCADPy"})
            if created.get("ok"):
                return None
            return created
        return self._error(
            "Attached to FreeCAD but could not hydrate its active document: "
            f"{attached['warning']}"
        )

    def _require_app(self) -> Optional[Dict[str, Any]]:
        if self.app is None:
            if self.execution_mode == "gui":
                return self._error(
                    "No desktop CAD application is selected. Call "
                    "list_cad_applications and select_cad_application; do not use "
                    "setup_backend or "
                    "execute_rapidcad_code for visible GUI edits."
                )
            return self._error(
                "No backend configured. Call setup_backend('freecad') first."
            )
        return None

    def _require_workplane(self) -> Optional[Dict[str, Any]]:
        ready = self._require_app()
        if ready is not None:
            return ready
        if self.active_workplane is None:
            return self._error("No active workplane. Call work_plane('XY') first.")
        return None

    def _ok(self, summary: str, **extra: Any) -> Dict[str, Any]:
        return {"ok": True, "summary": summary, **extra}

    def _error(self, message: str) -> Dict[str, Any]:
        return {"ok": False, "error": message}

    def _ensure_parent(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _resolve_export_path(self, path: str) -> Path:
        raw = Path(path).expanduser()
        if raw.is_absolute() or len(raw.parts) > 1:
            return raw.resolve()
        export_root = Path(
            os.environ.get("RAPIDCADPY_EXPORT_DIR", "/tmp/rapidcadpy_exports")
        ).expanduser()
        export_root.mkdir(parents=True, exist_ok=True)
        return (export_root / raw.name).resolve()
