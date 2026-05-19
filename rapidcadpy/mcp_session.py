"""Stateful RapidCADPy session used by the MCP server."""

from __future__ import annotations

import os
import json
import logging
import select
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .mcp_downloads import ensure_download_server, export_download_info, get_export_dir


@dataclass
class SemanticObject:
    id: str
    type: str
    label: str
    source_op: str
    source: str = "mcp_operation"
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


class RapidCADSession:
    """Small state store around one live RapidCADPy CAD document."""

    def __init__(self) -> None:
        self._worker: Optional[FreeCADWorkerClient] = None
        self.backend_name: Optional[str] = None
        self.app: Any = None
        self.active_workplane: Any = None
        self.active_workplane_id: Optional[str] = None
        self.active_shape_id: Optional[str] = None
        self.objects: Dict[str, SemanticObject] = {}
        self.runtime_objects: Dict[str, Any] = {}
        self.operations: list[OperationRecord] = []
        self.parameters: Dict[str, float] = {}
        self.geometry_signatures: Dict[str, Dict[str, Any]] = {}
        self._counters: Dict[str, int] = {}

    def setup_backend(
        self, cad_system: str = "freecad", document_name: str = "RapidCADPy_MCP"
    ) -> Dict[str, Any]:
        cad_system = cad_system.strip().lower()
        if cad_system not in {"freecad"}:
            return self._error(
                f"Unsupported backend '{cad_system}'. MVP supports only 'freecad'."
            )

        if self._worker is not None:
            self._worker.close()
            self._worker = None

        previous_logging_disable = logging.root.manager.disable
        try:
            if os.environ.get("RAPIDCADPY_MCP_WORKER") != "1":
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
            if os.environ.get("RAPIDCADPY_MCP_WORKER") == "1":
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

    def new_document(self, name: str = "RapidCADPy_MCP") -> Dict[str, Any]:
        if self._worker is not None:
            return self._worker.call("new_document", {"name": name})
        if self.app is None:
            return self._error(
                "No backend configured. Call setup_backend('freecad') first."
            )
        if self.backend_name != "freecad":
            return self._error("new_document MVP supports only FreeCAD backend.")

        try:
            from rapidcadpy.integrations.freecad.app import FreeCADApp

            self.app = FreeCADApp(doc_name=name)
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
        op_id = self._record(
            "new_document", {"name": name}, [], f"Created document '{name}'"
        )
        return self._ok(summary=f"Created document '{name}'", operation_id=op_id)

    def work_plane(
        self, plane: str = "XY", offset: Optional[float] = None
    ) -> Dict[str, Any]:
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

    def download_server_info(self) -> Dict[str, Any]:
        if self._worker is not None:
            return self._worker.call("download_server_info", {})
        try:
            info = ensure_download_server()
        except OSError as exc:
            return self._error(f"Could not start download server: {exc}")
        return self._ok(
            summary=f"Download server available at {info['base_url']}",
            **info,
        )

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
        try:
            doc = App.openDocument(str(file_path))
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
            if doc is not None:
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
        if self._worker is not None:
            return self._worker.call(
                "render",
                {
                    "path": path,
                    "view": view,
                    "shape_id": shape_id,
                    "width": width,
                    "height": height,
                },
            )
        shape_id = shape_id or self.active_shape_id
        if not shape_id:
            return self._error("No shape_id provided and no active shape exists.")
        shape = self.runtime_objects.get(shape_id)
        if shape is None:
            return self._error(f"Unknown shape_id '{shape_id}'")
        try:
            self._ensure_parent(path)
            shape.to_png(path, view=view, width=int(width), height=int(height))
        except Exception as exc:
            return self._error(f"render failed: {type(exc).__name__}: {exc}")
        op_id = self._record(
            "render",
            {"path": path, "view": view, "shape_id": shape_id},
            [],
            f"Rendered {shape_id} to {path}",
        )
        return self._ok(
            summary=f"Rendered {shape_id} to {path}",
            operation_id=op_id,
            path=os.path.abspath(path),
        )

    def add_parameter(
        self, name: str, value: float, units: str = "mm"
    ) -> Dict[str, Any]:
        if self._worker is not None:
            return self._worker.call(
                "add_parameter", {"name": name, "value": value, "units": units}
            )
        ready = self._require_app()
        if ready is not None:
            return ready
        try:
            stored = self.app.add_parameter(name, float(value), units=units)
        except Exception as exc:
            return self._error(f"add_parameter failed: {type(exc).__name__}: {exc}")
        self.parameters[name] = float(stored)
        op_id = self._record(
            "add_parameter",
            {"name": name, "value": value, "units": units},
            [],
            f"Added parameter {name}={stored} {units}",
        )
        return self._ok(
            summary=f"Added parameter {name}={stored} {units}",
            operation_id=op_id,
            parameters=dict(self.parameters),
        )

    def describe_state(self) -> Dict[str, Any]:
        if self._worker is not None:
            return self._worker.call("describe_state", {})
        return self._ok(
            summary=f"{len(self.objects)} objects, {len(self.operations)} operations",
            backend=self.backend_name,
            active_workplane_id=self.active_workplane_id,
            active_shape_id=self.active_shape_id,
            parameters=dict(self.parameters),
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
        download = export_download_info(str(resolved_path))
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
            **download,
        )

    def _geometry_signature(self, shape: Any) -> Dict[str, Any]:
        obj = getattr(shape, "obj", None)
        if obj is None:
            return {}
        return self._shape_signature(obj)

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

    def _require_app(self) -> Optional[Dict[str, Any]]:
        if self.app is None:
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
        return (get_export_dir() / raw.name).resolve()


class FreeCADWorkerClient:
    """JSON-lines client for a persistent FreeCAD Python worker."""

    def __init__(self) -> None:
        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        worker_path = os.path.join(package_root, "mcp", "freecad_worker.py")
        freecad_python = os.environ.get(
            "FREECAD_PYTHON",
            "/Applications/FreeCAD.app/Contents/Resources/bin/python",
        )
        env = os.environ.copy()
        env["RAPIDCADPY_MCP_WORKER"] = "1"
        env.setdefault(
            "RAPIDCADPY_MCP_WORKER_LOG", "/tmp/rapidcadpy_freecad_worker.log"
        )
        env.setdefault(
            "FREECAD_LIB_PATH", "/Applications/FreeCAD.app/Contents/Resources/lib"
        )
        env["PYTHONPATH"] = (
            package_root
            if not env.get("PYTHONPATH")
            else package_root + os.pathsep + env["PYTHONPATH"]
        )
        self.proc = subprocess.Popen(
            [freecad_python, worker_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        self.timeout_seconds = float(
            os.environ.get("RAPIDCADPY_MCP_WORKER_TIMEOUT", "120")
        )

    def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.proc.poll() is not None:
            return self._dead_response()
        if self.proc.stdin is None or self.proc.stdout is None:
            return {"ok": False, "error": "FreeCAD worker pipes unavailable."}

        request = {"method": method, "params": params}
        try:
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
            ready, _, _ = select.select(
                [self.proc.stdout], [], [], self.timeout_seconds
            )
            if not ready:
                return {
                    "ok": False,
                    "error": (
                        f"FreeCAD worker timed out after {self.timeout_seconds:g}s "
                        f"while running '{method}'. Check "
                        f"{os.environ.get('RAPIDCADPY_MCP_WORKER_LOG', '/tmp/rapidcadpy_freecad_worker.log')}."
                    ),
                }
            line = self.proc.stdout.readline()
        except Exception as exc:
            return {"ok": False, "error": f"FreeCAD worker call failed: {exc}"}

        if not line:
            return self._dead_response()
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            return {
                "ok": False,
                "error": f"FreeCAD worker returned non-JSON output: {exc}: {line[:500]}",
            }

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.call("shutdown", {})
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _dead_response(self) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": (
                f"FreeCAD worker exited with code {self.proc.returncode}. Check "
                f"{os.environ.get('RAPIDCADPY_MCP_WORKER_LOG', '/tmp/rapidcadpy_freecad_worker.log')}."
            ),
        }


def ensure_package_import_path() -> None:
    """Allow running vendor/rapidcadpy/mcp/server.py without pip install -e."""
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
