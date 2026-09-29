"""Focused DocumentService operations composed by :class:`CadSession`."""

from __future__ import annotations

import base64
import io
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional

from .cad_objects import CadObject
from .session_service import SessionService


class DocumentService(SessionService):
    def use_active_document(self) -> Dict[str, Any]:
        """Hydrate the document currently active in an attached FreeCAD GUI."""
        if self._live_backend is not None:
            result = self._live_backend.hydration.hydrate_active_document()
            if result.get("ok"):
                self.document = dict(result.get("document", {}))
                self.document_revision = result.get("document_revision")
                result.update(self._live_backend.application.to_dict())
            return result
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
            import FreeCAD as App

            from rapidcadpy.integrations.freecad.app import FreeCADApp

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
            return self._error(f"use_active_document failed: {type(exc).__name__}: {exc}")

    def new_document(self, name: str = "RapidCADPy") -> Dict[str, Any]:
        ready = self._ensure_gui_session(require_document=False)
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call("new_document", {"name": name})
        if self.app is None and self.execution_mode != "embedded":
            return self._error("No backend configured. Call setup_backend('freecad') first.")
        if self.backend_name not in {None, "freecad"}:
            return self._error("new_document MVP supports only FreeCAD backend.")

        try:
            if self._app_factory is not None:
                self.app = self._app_factory(name)
            else:
                from rapidcadpy.integrations.freecad.app import FreeCADApp

                self.app = FreeCADApp(doc_name=name)
            self.backend_name = "freecad"
        except Exception as exc:
            return self._error(f"Could not create document: {type(exc).__name__}: {exc}")

        self.active_workplane = None
        self.active_workplane_id = None
        self.active_shape_id = None
        self.objects.clear()
        self.runtime_objects.clear()
        self._shape_wrappers.clear()
        self.profile_references.clear()
        self.path_references.clear()
        self._profile_cache.clear()
        self.geometry_signatures.clear()
        self.document = {
            "name": getattr(self.app.get_doc(), "Name", name),
            "label": getattr(self.app.get_doc(), "Label", name),
            "file_name": getattr(self.app.get_doc(), "FileName", ""),
        }
        self.cad_document = getattr(self.app, "cad_document", None)
        self.parameters.clear()
        self._recalculate_document_revision()
        op_id = self._record("new_document", {"name": name}, [], f"Created document '{name}'")
        return self._ok(
            summary=f"Created document '{name}'",
            operation_id=op_id,
            document_revision=self.document_revision,
        )

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
            return self._error(f"Could not hydrate opened document: {type(exc).__name__}: {exc}")

    def execute_code(self, code: str, allow_direct_geometry: bool = False) -> Dict[str, Any]:
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
            return self._error("execute_code requires an attached or embedded FreeCAD session.")

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
        existing_files = {path.resolve() for path in execution_dir.iterdir() if path.is_file()}
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
        original_active_name = str(getattr(getattr(App, "ActiveDocument", None), "Name", ""))
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
                active_path = Path(active_file_name).resolve() if active_file_name else None
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
            return self._error(f"describe_freecad_file failed: {type(exc).__name__}: {exc}")
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
                render_worker = Path(__file__).resolve().parent / "workers" / "render_stl_worker.py"
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
                    f"render failed in worker and server fallback: {type(exc).__name__}: {exc}"
                )
            finally:
                temp_stl.unlink(missing_ok=True)
            return self._render_response(resolved_path, view, shape_id, int(width), int(height))

        shape_id = shape_id or self.active_shape_id
        if not shape_id:
            return self._error("No shape_id provided and no active shape exists.")
        shape = self.runtime_objects.get(shape_id)
        if shape is None:
            return self._error(f"Unknown shape_id '{shape_id}'")
        try:
            self._ensure_parent(str(resolved_path))
            shape.to_png(str(resolved_path), view=view, width=int(width), height=int(height))
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
            for value in self.cad_document.feature_definitions_for_native_names(native_names)
        ]

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
                f"Could not set {object_id}.{property_name}: {type(exc).__name__}: {exc}"
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
        """Select a native object in the connected CAD GUI."""
        if self._live_backend is not None:
            return self._live_backend.connection.call("select_object", {"object_id": object_id})
        if self._gui_connection is None:
            return self._error("Object selection requires FreeCAD GUI mode.")
        return self._gui_connection.call("select_object", {"object_id": object_id})

    def fit_view(self) -> Dict[str, Any]:
        """Fit the connected CAD GUI view to the active model."""
        if self._live_backend is not None:
            return self._live_backend.connection.call("fit_view", {})
        if self._gui_connection is None:
            return self._error("fit_view requires FreeCAD GUI mode.")
        return self._gui_connection.call("fit_view", {})

    def export_screenshot(
        self,
        path: Optional[str] = None,
        view: str = "isometric",
        width: int = 1024,
        height: int = 768,
        fit: bool = True,
    ) -> Dict[str, Any]:
        """Capture the active native viewport and optionally write it as PNG."""

        ready = self._ensure_gui_session(create_document_if_missing=False)
        if ready is not None:
            return ready
        if self._worker is not None:
            result = self._worker.call(
                "export_screenshot",
                {
                    "path": None,
                    "view": view,
                    "width": width,
                    "height": height,
                    "fit": fit,
                },
            )
            if not result.get("ok") or path is None:
                return result
            encoded = result.pop("image_base64", None)
            if not isinstance(encoded, str) or not encoded:
                return self._error("CAD viewport backend returned no screenshot image data.")
            try:
                content = base64.b64decode(encoded, validate=True)
                resolved_path = self._write_screenshot(path, content)
            except Exception as exc:
                return self._error(
                    f"Could not write viewport screenshot: {type(exc).__name__}: {exc}"
                )
            result.update(
                {
                    "summary": f"Exported {result.get('view', view)} viewport screenshot",
                    "path": str(resolved_path),
                    "filename": resolved_path.name,
                    "size": len(content),
                }
            )
            return result

        if self.cad_document is None:
            return self._error("No native CAD document is open.")
        try:
            from .viewport import (
                create_viewport_backend,
                normalize_viewport_name,
                validate_viewport_size,
            )

            normalized_view = normalize_viewport_name(view)
            normalized_width, normalized_height = validate_viewport_size(
                width,
                height,
            )
            screenshot = create_viewport_backend(self.cad_document).capture_screenshot(
                document=self.cad_document,
                view=normalized_view,
                width=normalized_width,
                height=normalized_height,
                fit=bool(fit),
            )
        except Exception as exc:
            return self._error(
                f"Could not capture viewport screenshot: {type(exc).__name__}: {exc}"
            )

        op_id = self._record(
            "export_screenshot",
            {
                "path": path,
                "view": screenshot.view,
                "width": screenshot.width,
                "height": screenshot.height,
                "fit": bool(fit),
            },
            [],
            f"Captured {screenshot.view} viewport screenshot",
        )
        result = self._ok(
            summary=f"Captured {screenshot.view} viewport screenshot",
            operation_id=op_id,
            media_type=screenshot.media_type,
            width=screenshot.width,
            height=screenshot.height,
            view=screenshot.view,
            fit=bool(fit),
            size=len(screenshot.content),
            document_revision=self.document_revision,
        )
        if path is None:
            result["image_base64"] = base64.b64encode(screenshot.content).decode("ascii")
            return result
        try:
            resolved_path = self._write_screenshot(path, screenshot.content)
        except Exception as exc:
            return self._error(f"Could not write viewport screenshot: {type(exc).__name__}: {exc}")
        result.update(
            {
                "summary": f"Exported {screenshot.view} viewport screenshot",
                "path": str(resolved_path),
                "filename": resolved_path.name,
            }
        )
        return result

    @staticmethod
    def _write_screenshot(path: str, content: bytes) -> Path:
        resolved_path = Path(path).expanduser().resolve()
        if resolved_path.suffix.lower() != ".png":
            raise ValueError("Viewport screenshot path must use the .png extension.")
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_bytes(content)
        return resolved_path

    def get_history(self, limit: int = 50) -> Dict[str, Any]:
        if self._worker is not None:
            return self._worker.call("get_history", {"limit": limit})
        records = self.operations[-int(limit) :]
        return self._ok(
            summary=f"{len(records)} operations returned",
            operations=[op.to_dict() for op in records],
        )
