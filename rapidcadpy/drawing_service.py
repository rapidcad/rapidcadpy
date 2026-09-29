"""Focused DrawingService operations composed by :class:`CadSession`."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .cad_objects import CadObject
from .session_service import SessionService


class DrawingService(SessionService):
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
                        "angle_degrees": feature.get("countersink_angle_degrees")
                    },
                }
            )
        return requests

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
                return self._error(f"Object '{object_id}' is not a live native CAD object.")
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
            return self._error(f"Could not generate technical drawing: {type(exc).__name__}: {exc}")

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
        view_choices = (
            ", ".join(f"{name}={object_id}" for name, object_id in view_lookup.items()) or "none"
        )
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
            return self._error(f"Could not inspect drawing: {type(exc).__name__}: {exc}")
        drawing["views"] = inspection["views"]
        drawing["items"] = inspection["items"]
        view_lookup = self._drawing_view_lookup(inspection["views"])
        view_choices = ", ".join(f"{name}={object_id}" for name, object_id in view_lookup.items())
        if not view_choices:
            view_choices = (
                ", ".join(
                    f"{view.get('native_name', 'view')}={view.get('id')}"
                    for view in inspection["views"]
                )
                or "none"
            )
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
                standard=str((drawing.get("metadata") or {}).get("standard", "ISO")),
            )
        except Exception as exc:
            return self._error(f"Could not add feature dimension: {type(exc).__name__}: {exc}")
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
                anchor_mm=(self._drawing_position(anchor_mm) if anchor_mm is not None else None),
                elbow_mm=(self._drawing_position(elbow_mm) if elbow_mm is not None else None),
                text_position_mm=(
                    self._drawing_position(text_position_mm)
                    if text_position_mm is not None
                    else None
                ),
            )
        except Exception as exc:
            return self._error(f"Could not add drawing leader: {type(exc).__name__}: {exc}")
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
            return self._error(f"Could not move drawing item: {type(exc).__name__}: {exc}")
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
            return self._error(f"Could not export current drawing: {type(exc).__name__}: {exc}")
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
        if not resolved_id or resolved_id not in self.drawings or resolved_id not in discovered_ids:
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
            return self._error(f"Could not discover drawing pages: {type(exc).__name__}: {exc}")
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
        return {name: object_ids[0] for name, object_ids in grouped.items() if len(object_ids) == 1}

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
                if str(view.get("view_name") or "").strip().lower() == normalized_name
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
                str(view.get("view_name") or view.get("native_name") or "view"): str(view.get("id"))
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
            ids_by_name[name] for name in edit.created_native_names if name in ids_by_name
        ]
        changed_ids = [
            ids_by_name[name] for name in edit.changed_native_names if name in ids_by_name
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
                "next_action": ("Call export_current_drawing to refresh the downloadable PDF."),
            }
        )
        return result
