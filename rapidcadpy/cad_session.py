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
from typing import Any, Dict, Optional

from .cad_objects import CadDocument, CadFeature, CadObject, CadParameter
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
        "feature.extrude",
        "object.list",
        "object.inspect",
        "object.set_property",
        "parameter.list",
        "parameter.create",
        "parameter.update",
        "parameter.bind",
        "drawing.generate",
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
        self.operations: list[OperationRecord] = []
        self.parameters: Dict[str, CadParameter] = {}
        self.geometry_signatures: Dict[str, Dict[str, Any]] = {}
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

    def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute RapidCADPy/FreeCAD code inside the attached GUI process."""
        ready = self._ensure_gui_session(require_document=False)
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call("execute_code", {"code": code})
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
        previous_cwd = Path.cwd()
        try:
            os.chdir(execution_dir)
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(compile(normalized, "<rapidcadpy-live>", "exec"), namespace)
        except Exception as exc:
            return self._error(
                "Live FreeCAD code failed: "
                f"{type(exc).__name__}: {exc}\n"
                f"stdout:\n{stdout_buffer.getvalue()}\n"
                f"stderr:\n{stderr_buffer.getvalue()}"
            )
        finally:
            os.chdir(previous_cwd)

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
        self.runtime_objects[shape_id] = shape
        if getattr(shape, "feature", None) is not None:
            shape_id = shape.feature.document.bind_object_id(
                shape.feature.native_name,
                shape_id,
            )
            shape.feature.id = shape_id
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
        if target is None:
            return self._error(f"Unknown target_id '{target_id}'")
        if tool is None:
            return self._error(f"Unknown tool_id '{tool_id}'")
        try:
            result = target.cut(tool)
        except Exception as exc:
            return self._error(f"cut failed: {type(exc).__name__}: {exc}")

        result_id = target_id
        self.runtime_objects[result_id] = result
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
        shape = self.runtime_objects.get(shape_id)
        if shape is None:
            return self._error(f"Unknown shape_id '{shape_id}'")
        try:
            if selector:
                result = shape.edges(selector).fillet(float(radius))
            else:
                result = shape.fillet(float(radius))
        except Exception as exc:
            return self._error(f"fillet failed: {type(exc).__name__}: {exc}")

        self.runtime_objects[shape_id] = result
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
            objects=[obj.to_dict() for obj in self.objects.values()],
        )

    def get_object(self, object_id: str) -> Dict[str, Any]:
        """Return one semantic object by its RapidCAD ID."""
        if self._worker is not None:
            return self._worker.call("get_object", {"object_id": object_id})
        semantic_object = self.objects.get(object_id)
        if semantic_object is None:
            return self._error(f"Unknown object_id '{object_id}'")
        return self._ok(
            summary=f"Object {object_id}",
            object=semantic_object.to_dict(),
            document_revision=self.document_revision,
        )

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
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a linked native drawing and export a print-ready PDF."""

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
        result = drawing.to_dict()
        result.update(
            {
                "ok": True,
                "summary": (
                    f"Created {standard.strip().upper()} "
                    f"{sheet_size.strip().upper()} {projection_angle.strip().lower()}-angle "
                    f"technical drawing for {resolved_part_name}"
                ),
                "run_id": resolved_run_id,
                "created_object_ids": created_object_ids,
                "document_revision": self.document_revision,
            }
        )
        return result

    def save_document(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Save the active native document, optionally to a new path."""
        if self._worker is not None:
            return self._worker.call("save_document", {"path": path})
        if self.cad_document is None:
            return self._error("No native CAD document is open.")
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
                shape = self.runtime_objects.get(shape_id)
                if shape is None:
                    return self._error(f"Unknown shape_id '{shape_id}'")
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
