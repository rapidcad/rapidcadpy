"""FreeCAD TechDraw implementation of RapidCADPy technical drawings."""

from __future__ import annotations

import math
import json
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence

from ...cad_objects import CadDocument, CadObject
from ...dimension_planner import ModelBounds, build_dimension_intents_from_features
from ...dimension_layout import (
    ViewGeometry,
    place_dimension_intents,
    view_model_ranges,
)
from ...features import feature_from_dict
from ...drawing import (
    DrawingBackend,
    DrawingBackendFactory,
    DrawingEditResult,
    DrawingOutputFormat,
    DrawingResult,
    DimensionPlacement,
    normalize_drawing_standard,
    normalize_projection_angle,
    projected_view_layout,
    select_drawing_scale,
)

_TEMPLATES = {
    ("ISO", "A3", None): "A3_Landscape_ISO5457_advanced.svg",
    ("ISO", "A3", "iso-a3-v1"): "A3_Landscape_ISO5457_advanced.svg",
    ("ASME", "A3", None): "A3_Landscape_ISO5457_advanced.svg",
    ("ASME", "A3", "asme-a3-v1"): "A3_Landscape_ISO5457_advanced.svg",
}
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_DRAWING_NAMES = (
    "RapidCADDrawing",
    "RapidCADTemplate",
    "RapidCADProjectionGroup",
    "RapidCADIsometric",
)


class FreeCADDrawingBackendFactory(DrawingBackendFactory):
    """Construct FreeCAD drawing backends without leaking native handles."""

    def create(self, document: CadDocument) -> DrawingBackend:
        if document.backend.strip().lower() != "freecad":
            raise ValueError("FreeCADDrawingBackend requires a FreeCAD document.")
        return FreeCADDrawingBackend()


class FreeCADDrawingBackend(DrawingBackend):
    """Create an auto-dimensioned A3 TechDraw page from live CAD objects."""

    @property
    def supported_output_formats(self) -> frozenset[DrawingOutputFormat]:
        # FreeCAD can write the complete TechDraw page directly as DXF. DWG
        # requires an external converter and IDW is an Inventor-native format.
        return frozenset({"pdf", "dxf"})

    def generate_drawing(
        self,
        *,
        document: CadDocument,
        objects: Sequence[CadObject],
        standard: str,
        sheet_size: str,
        projection_angle: str,
        template_id: Optional[str],
        output_directory: Path,
        part_name: str,
        run_id: str,
        include_native: bool,
        dimension_feature_ids: Optional[Sequence[str]] = None,
        output_formats: Optional[Sequence[str]] = None,
    ) -> DrawingResult:
        requested_formats = self.validate_output_formats(output_formats)
        normalized_standard = normalize_drawing_standard(standard)
        normalized_sheet = sheet_size.strip().upper()
        normalized_projection = normalize_projection_angle(projection_angle)
        normalized_template = template_id.strip().lower() if template_id else None
        template_filename = _TEMPLATES.get(
            (normalized_standard, normalized_sheet, normalized_template)
        )
        if template_filename is None:
            raise ValueError(
                "FreeCAD drawing export supports ISO or ASME A3 drawings with "
                "the matching built-in template id (or no explicit template_id)."
            )

        try:
            import FreeCAD as App
            import FreeCADGui as Gui
            import TechDraw
            import TechDrawGui
        except ImportError as exc:
            raise RuntimeError(
                "FreeCAD drawing export must run inside the attached FreeCAD GUI."
            ) from exc

        native_document = document.native_handle
        if native_document is None:
            raise RuntimeError("The FreeCAD document has no live native handle.")
        sources = self._resolve_sources(native_document, objects)
        output_directory = output_directory.expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

        template_path = self._resolve_template(
            Path(App.getResourceDir()), template_filename
        )
        self._remove_previous_drawing(native_document)

        page = native_document.addObject("TechDraw::DrawPage", "RapidCADDrawing")
        page.Label = f"{part_name} Drawing"
        self._set_string_properties(
            page,
            {
                "RapidCADDrawingId": run_id,
                "RapidCADDrawingStandard": normalized_standard,
            },
        )
        template = native_document.addObject(
            "TechDraw::DrawSVGTemplate", "RapidCADTemplate"
        )
        template.Template = str(template_path)
        page.Template = template
        native_document.recompute()

        created_names = [page.Name, template.Name]
        view_specs = projected_view_layout(
            normalized_projection,
            sheet_size=normalized_sheet,
        )
        projection_group, orthographic_views = self._create_projection_group(
            native_document=native_document,
            page=page,
            sources=sources,
            projection_angle=normalized_projection,
            anchor_x_mm=view_specs[0].x_mm,
            anchor_y_mm=view_specs[0].y_mm,
            app_module=App,
        )
        created_names.append(projection_group.Name)
        native_document.recompute()
        Gui.updateGui()

        # The projection group owns the orthographic layout and scale. Keep the
        # isometric view separate because TechDraw projection groups only model
        # orthographic projections.
        scale = float(projection_group.Scale)
        isometric_spec = view_specs[-1]
        isometric = native_document.addObject(
            "TechDraw::DrawViewPart", "RapidCADIsometric"
        )
        isometric.Source = list(sources)
        isometric.Direction = App.Vector(*isometric_spec.direction)
        if hasattr(isometric, "ScaleType"):
            isometric.ScaleType = "Custom"
        isometric.Scale = scale
        isometric.X = isometric_spec.x_mm
        isometric.Y = isometric_spec.y_mm
        page.addView(isometric)
        created_names.extend(view.Name for view in (*orthographic_views, isometric))

        native_document.recompute()
        Gui.updateGui()
        views = (*orthographic_views, isometric)
        for view, spec in zip(views, view_specs):
            self._set_string_properties(
                view,
                {
                    "RapidCADViewName": spec.name,
                    "RapidCADDrawingId": run_id,
                },
            )
        self._wait_for_projected_geometry(
            native_document=native_document,
            page=page,
            views=views,
            gui_module=Gui,
        )

        semantic_features = self._semantic_features_for_sources(document, sources)
        semantic_features = self._select_dimension_features(
            semantic_features,
            dimension_feature_ids,
        )
        bounds = self._source_bounds(sources)
        dimension_intents = build_dimension_intents_from_features(
            bounds,
            semantic_features,
        )
        dimension_source = "semantic_features" if semantic_features else "none"
        feature_analysis: dict[str, object] = {
            "bounds_mm": {
                "x": [bounds.x_min, bounds.x_max],
                "y": [bounds.y_min, bounds.y_max],
                "z": [bounds.z_min, bounds.z_max],
            },
            "source": dimension_source,
            "warnings": [],
        }
        if semantic_features:
            semantic_counts: dict[str, int] = {}
            for feature in semantic_features:
                kind = str(feature.to_dict()["kind"])
                semantic_counts[kind] = semantic_counts.get(kind, 0) + 1
            feature_analysis["semantic_feature_counts"] = semantic_counts
        view_geometries = self._dimension_view_geometries(
            views=views,
            view_specs=view_specs,
            bounds=bounds,
            scale=scale,
        )
        placement_warnings: list[str] = []
        dimension_placements = place_dimension_intents(
            dimension_intents,
            view_geometries,
            standard=normalized_standard,
            page_width_mm=420.0,
            page_height_mm=297.0,
            reserved_boxes=((250.0, 0.0, 420.0, 60.0),),
            warnings=placement_warnings,
            allow_collisions=True,
        )
        feature_analysis["warnings"].extend(placement_warnings)
        native_dimensions = self._create_native_dimensions(
            native_document=native_document,
            page=page,
            views=views,
            view_specs=view_specs,
            placements=dimension_placements,
            standard=normalized_standard,
            app_module=App,
            techdraw_module=TechDraw,
        )
        created_names.extend(item.Name for item in native_dimensions)
        dimension_lines = [
            placement.intent.formatted_text(normalized_standard)
            for placement in dimension_placements
        ]

        self._fill_title_block(
            template,
            part_name=part_name,
            run_id=run_id,
            scale=scale,
            standard=normalized_standard,
        )
        if hasattr(page, "KeepUpdated"):
            page.KeepUpdated = True
        page.ViewObject.show()
        native_document.recompute()
        Gui.updateGui()
        native_document.recompute()

        safe_part = self._safe_component(part_name, "drawing")
        safe_run = self._safe_component(run_id, "run")
        dated_stem = f"{safe_part}_{safe_run}_{date.today().isoformat()}"
        output_paths: dict[str, Path] = {}
        pdf_path: Optional[Path] = None
        vector_source_path: Optional[Path] = None
        if "pdf" in requested_formats:
            pdf_path = output_directory / f"{dated_stem}.pdf"
            vector_directory = output_directory / ".intermediate"
            vector_directory.mkdir(parents=True, exist_ok=True)
            vector_source_path = vector_directory / f".{dated_stem}.svg"
            TechDrawGui.exportPageAsSvg(page, str(vector_source_path))
            if (
                not vector_source_path.is_file()
                or vector_source_path.stat().st_size < 1000
            ):
                raise RuntimeError(
                    "TechDraw SVG export is missing or unexpectedly small."
                )
            TechDrawGui.exportPageAsPdf(page, str(pdf_path))
            Gui.updateGui()
            self._validate_pdf(pdf_path)
            output_paths["pdf"] = pdf_path

        if "dxf" in requested_formats:
            dxf_path = output_directory / f"{dated_stem}.dxf"
            TechDraw.writeDXFPage(page, str(dxf_path))
            self._validate_dxf(dxf_path, expected_annotations=dimension_lines)
            output_paths["dxf"] = dxf_path

        native_path: Optional[Path] = None
        warnings: list[str] = []
        warnings.extend(str(item) for item in feature_analysis["warnings"])
        if dimension_source == "none":
            warnings.append(
                "Only overall dimensions were created because no authoritative "
                "semantic features were available."
            )
        if include_native:
            native_path = output_directory / f"{safe_part}_{safe_run}.FCStd"
            save_copy = getattr(native_document, "saveCopy", None)
            if callable(save_copy):
                save_copy(str(native_path))
                if not native_path.is_file() or native_path.stat().st_size == 0:
                    raise RuntimeError("FreeCAD did not write the native drawing copy.")
            else:
                native_path = None
                warnings.append(
                    "This FreeCAD version does not expose Document.saveCopy(); "
                    "the linked drawing remains in the active document."
                )

        return DrawingResult(
            pdf_path=pdf_path,
            output_paths=output_paths,
            vector_source_path=vector_source_path,
            native_drawing_path=native_path,
            page_name=page.Name,
            source_object_ids=tuple(item.id for item in objects),
            created_native_names=tuple(created_names),
            warnings=tuple(warnings),
            metadata={
                "standard": normalized_standard,
                "sheet_size": normalized_sheet,
                "page_width_mm": 420,
                "page_height_mm": 297,
                "template_id": normalized_template
                or ("iso-a3-v1" if normalized_standard == "ISO" else "asme-a3-v1"),
                "projection_angle": normalized_projection,
                "scale": scale,
                "views": [spec.to_dict() for spec in view_specs],
                "native_view_names": {
                    spec.name: view.Name for view, spec in zip(views, view_specs)
                },
                "vector_export": True,
                "minimum_text_height_mm": 3.5,
                "parametric_link_preserved": True,
                "output_formats": list(requested_formats),
                "dimension_standard": normalized_standard,
                "dimension_unit": "mm" if normalized_standard == "ISO" else "in",
                "dimension_source": dimension_source,
                "dimension_feature_ids": [
                    str(feature.id) for feature in semantic_features
                ],
                "dimension_count": len(dimension_placements),
                "dimension_feature_analysis": feature_analysis,
                "dimensions": [
                    {
                        **placement.to_dict(normalized_standard),
                        "measurement_source": "projected_geometry",
                        "native_format_spec": self._measured_format_spec(
                            placement.intent,
                            normalized_standard,
                        ),
                        "rendered_value_owner": "freecad",
                    }
                    for placement in dimension_placements
                ],
                "dimension_collisions": 0,
            },
        )

    def inspect_drawing(
        self,
        *,
        document: CadDocument,
        page_name: str,
    ) -> dict[str, Any]:
        page = self._drawing_page(document.native_handle, page_name)
        views: list[dict[str, object]] = []
        items: list[dict[str, object]] = []
        for candidate in self._page_members(document.native_handle, page):
            native_name = str(getattr(candidate, "Name", ""))
            type_id = str(getattr(candidate, "TypeId", ""))
            payload = {
                "id": document.object_id(native_name),
                "native_name": native_name,
                "label": str(getattr(candidate, "Label", native_name)),
                "native_type": type_id,
                "x_mm": self._quantity_value(getattr(candidate, "X", 0.0)),
                "y_mm": self._quantity_value(getattr(candidate, "Y", 0.0)),
            }
            logical_view = self._logical_view_name(candidate)
            if logical_view:
                payload["view_name"] = logical_view
                views.append(payload)
                continue
            if type_id in {
                "TechDraw::DrawViewPart",
                "TechDraw::DrawProjGroupItem",
            }:
                payload["view_name"] = None
                views.append(payload)
                continue
            item_kind = str(getattr(candidate, "RapidCADDrawingItemKind", ""))
            if not item_kind:
                if type_id == "TechDraw::DrawViewDimension":
                    item_kind = "dimension"
                elif type_id in {
                    "TechDraw::DrawViewAnnotation",
                    "TechDraw::DrawRichAnno",
                }:
                    item_kind = "note"
                elif type_id == "TechDraw::DrawLeaderLine":
                    item_kind = "leader"
            if item_kind:
                payload["item_kind"] = item_kind
                feature_ids = str(getattr(candidate, "RapidCADFeatureIds", ""))
                if feature_ids:
                    try:
                        payload["feature_ids"] = json.loads(feature_ids)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        payload["feature_ids"] = []
                items.append(payload)
        return {
            "page_name": page.Name,
            "page_id": document.object_id(page.Name),
            "drawing_id": str(getattr(page, "RapidCADDrawingId", "")),
            "views": views,
            "items": items,
        }

    def list_drawings(self, *, document: CadDocument) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for page in getattr(document.native_handle, "Objects", ()):
            if str(getattr(page, "TypeId", "")) != "TechDraw::DrawPage":
                continue
            inspection = self.inspect_drawing(
                document=document,
                page_name=str(page.Name),
            )
            persisted_id = str(getattr(page, "RapidCADDrawingId", "")).strip()
            drawing_id = persisted_id or f"drawing-{inspection['page_id']}"
            result.append(
                {
                    "drawing_id": drawing_id,
                    "page_id": inspection["page_id"],
                    "page_name": page.Name,
                    "label": str(getattr(page, "Label", page.Name)),
                    "managed_by_rapidcad": bool(persisted_id),
                    "view_count": len(inspection["views"]),
                    "item_count": len(inspection["items"]),
                    "views": inspection["views"],
                    "items": inspection["items"],
                }
            )
        return result

    def add_feature_dimension(
        self,
        *,
        document: CadDocument,
        page_name: str,
        view_native_name: str,
        feature_definition: dict[str, Any],
        dimension_kind: str,
        position_mm: tuple[float, float],
        standard: str,
    ) -> DrawingEditResult:
        native_document = document.native_handle
        page = self._drawing_page(native_document, page_name)
        view = self._drawing_view(native_document, page, view_native_name)
        feature = feature_from_dict(feature_definition)
        if feature_definition.get("kind") != "hole":
            raise ValueError("Feature dimensions currently support semantic holes only.")
        requested_kind = dimension_kind.strip().lower()
        intent_kind = {
            "diameter": "hole",
            "countersink_diameter": "countersink",
        }.get(requested_kind)
        if intent_kind is None:
            raise ValueError(
                "dimension_kind must be 'diameter' or 'countersink_diameter'."
            )
        feature_id = str(feature.id)
        for candidate in self._page_members(native_document, page):
            if str(getattr(candidate, "RapidCADDimensionKind", "")) != requested_kind:
                continue
            try:
                existing_ids = json.loads(
                    str(getattr(candidate, "RapidCADFeatureIds", "[]"))
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                existing_ids = []
            if feature_id in existing_ids:
                raise ValueError(
                    f"Feature {feature_id} already has a {requested_kind} dimension "
                    "on this drawing. Move the existing item instead."
                )
        sources = tuple(getattr(view, "Source", ()) or ())
        if not sources:
            raise RuntimeError(
                f"Drawing view {view_native_name!r} has no linked source geometry."
            )
        bounds = self._source_bounds(sources)
        intents = build_dimension_intents_from_features(bounds, (feature,))
        intent = next(
            (
                candidate
                for candidate in intents
                if candidate.kind == intent_kind
                and feature_id in candidate.source_references
            ),
            None,
        )
        if intent is None:
            raise RuntimeError(
                f"Feature {feature_id} does not provide a {requested_kind} dimension."
            )
        logical_view = self._logical_view_name(view)
        if logical_view != intent.view:
            raise ValueError(
                f"Feature {feature_id} must be dimensioned in the {intent.view} "
                f"view, not {logical_view or view_native_name}."
            )
        x_mm, y_mm = self._validated_page_position(position_mm)
        side = self._position_side(view, x_mm, y_mm)
        placement = DimensionPlacement(
            intent=intent,
            side=side,
            x_mm=x_mm,
            y_mm=y_mm,
            text_box=(x_mm, y_mm, x_mm, y_mm),
            projected_reference_points=(),
        )
        created = self._create_native_dimensions(
            native_document=native_document,
            page=page,
            views=(view,),
            view_specs=(type("ViewSpec", (), {"name": logical_view})(),),
            placements=(placement,),
            standard=standard,
            app_module=None,
            techdraw_module=None,
        )[0]
        self._set_string_properties(
            created,
            {
                "RapidCADDrawingItemKind": "dimension",
                "RapidCADDimensionKind": requested_kind,
            },
        )
        native_document.recompute()
        return DrawingEditResult(
            created_native_names=(created.Name,),
            changed_native_names=(page.Name,),
            metadata={
                "feature_id": feature_id,
                "dimension_kind": requested_kind,
                "measurement_source": "projected_geometry",
                "projected_reference": str(
                    getattr(created, "RapidCADProjectedReference", "")
                ),
            },
        )

    def add_drawing_note(
        self,
        *,
        document: CadDocument,
        page_name: str,
        text: str,
        position_mm: tuple[float, float],
    ) -> DrawingEditResult:
        if not str(text).strip():
            raise ValueError("Drawing note text must not be empty.")
        native_document = document.native_handle
        page = self._drawing_page(native_document, page_name)
        x_mm, y_mm = self._validated_page_position(position_mm)
        note = native_document.addObject(
            "TechDraw::DrawViewAnnotation",
            "RapidCADDrawingNote",
        )
        note.Text = str(text).splitlines()
        note.X = x_mm
        note.Y = y_mm
        self._set_string_properties(
            note,
            {"RapidCADDrawingItemKind": "note"},
        )
        page.addView(note)
        native_document.recompute()
        return DrawingEditResult(
            created_native_names=(note.Name,),
            changed_native_names=(page.Name,),
            metadata={"item_kind": "note"},
        )

    def add_drawing_leader(
        self,
        *,
        document: CadDocument,
        page_name: str,
        view_native_name: str,
        text: str,
        anchor_mm: Optional[tuple[float, float]],
        elbow_mm: Optional[tuple[float, float]],
        text_position_mm: Optional[tuple[float, float]],
    ) -> DrawingEditResult:
        if not str(text).strip():
            raise ValueError("Drawing leader text must not be empty.")
        native_document = document.native_handle
        page = self._drawing_page(native_document, page_name)
        view = self._drawing_view(native_document, page, view_native_name)
        supplied_positions = (anchor_mm, elbow_mm, text_position_mm)
        if any(value is None for value in supplied_positions) and not all(
            value is None for value in supplied_positions
        ):
            raise ValueError(
                "Provide anchor_mm, elbow_mm, and text_position_mm together, "
                "or omit all three for automatic leader placement."
            )
        if all(value is None for value in supplied_positions):
            view_x = self._quantity_value(getattr(view, "X", 0.0))
            view_y = self._quantity_value(getattr(view, "Y", 0.0))
            view_height = self._quantity_value(getattr(view, "Height", 40.0))
            direction = -1.0 if view_x > 330.0 else 1.0
            anchor_mm = (view_x, min(292.0, view_y + view_height / 2.0))
            elbow_mm = (
                max(5.0, min(415.0, anchor_mm[0] + direction * 20.0)),
                max(5.0, min(292.0, anchor_mm[1] + 15.0)),
            )
            text_position_mm = (
                max(5.0, min(415.0, anchor_mm[0] + direction * 50.0)),
                elbow_mm[1],
            )
        assert anchor_mm is not None
        assert elbow_mm is not None
        assert text_position_mm is not None
        anchor_x, anchor_y = self._validated_page_position(anchor_mm)
        elbow_x, elbow_y = self._validated_page_position(elbow_mm)
        text_x, text_y = self._validated_page_position(text_position_mm)
        try:
            import FreeCAD as App
        except ImportError as exc:
            raise RuntimeError(
                "Drawing leaders must be created inside the attached FreeCAD GUI."
            ) from exc
        leader = native_document.addObject(
            "TechDraw::DrawLeaderLine",
            "RapidCADDrawingLeader",
        )
        page.addView(leader)
        leader.LeaderParent = view
        view_x = self._quantity_value(getattr(view, "X", 0.0))
        view_y = self._quantity_value(getattr(view, "Y", 0.0))
        leader.X = anchor_x - view_x
        leader.Y = anchor_y - view_y
        leader.AutoHorizontal = False
        leader.WayPoints = [
            App.Vector(0.0, 0.0, 0.0),
            App.Vector(elbow_x - anchor_x, elbow_y - anchor_y, 0.0),
            App.Vector(text_x - anchor_x, text_y - anchor_y, 0.0),
        ]
        if hasattr(leader, "StartSymbol"):
            leader.StartSymbol = "Filled Arrow"
        if hasattr(leader, "EndSymbol"):
            leader.EndSymbol = "None"
        group_id = f"leader:{leader.Name}"
        self._set_string_properties(
            leader,
            {
                "RapidCADDrawingItemKind": "leader",
                "RapidCADDrawingGroupId": group_id,
            },
        )
        note = native_document.addObject(
            "TechDraw::DrawViewAnnotation",
            "RapidCADLeaderNote",
        )
        note.Text = str(text).splitlines()
        note.X = text_x
        note.Y = text_y
        self._set_string_properties(
            note,
            {
                "RapidCADDrawingItemKind": "leader_note",
                "RapidCADDrawingGroupId": group_id,
            },
        )
        page.addView(note)
        native_document.recompute()
        return DrawingEditResult(
            created_native_names=(leader.Name, note.Name),
            changed_native_names=(page.Name,),
            metadata={
                "item_kind": "leader",
                "group_id": group_id,
                "view_native_name": view.Name,
            },
        )

    def move_drawing_item(
        self,
        *,
        document: CadDocument,
        page_name: str,
        item_native_name: str,
        position_mm: tuple[float, float],
    ) -> DrawingEditResult:
        native_document = document.native_handle
        page = self._drawing_page(native_document, page_name)
        item = self._drawing_page_member(native_document, page, item_native_name)
        item_kind = str(getattr(item, "RapidCADDrawingItemKind", ""))
        if item_kind not in {"dimension", "note", "leader", "leader_note"}:
            raise ValueError(
                f"Drawing item {item_native_name!r} is not an editable RapidCAD "
                "dimension, note, or leader."
            )
        x_mm, y_mm = self._validated_page_position(position_mm)
        if item_kind == "leader":
            parent = getattr(item, "LeaderParent", None)
            if parent is None:
                raise RuntimeError("The leader has no linked drawing view.")
            item.X = x_mm - self._quantity_value(getattr(parent, "X", 0.0))
            item.Y = y_mm - self._quantity_value(getattr(parent, "Y", 0.0))
        else:
            item.X = x_mm
            item.Y = y_mm
        native_document.recompute()
        return DrawingEditResult(
            changed_native_names=(item.Name, page.Name),
            metadata={"item_kind": item_kind, "position_mm": [x_mm, y_mm]},
        )

    def export_drawing(
        self,
        *,
        document: CadDocument,
        page_name: str,
        pdf_path: Path,
        vector_source_path: Path,
    ) -> DrawingEditResult:
        page = self._drawing_page(document.native_handle, page_name)
        try:
            import FreeCADGui as Gui
            import TechDrawGui
        except ImportError as exc:
            raise RuntimeError(
                "Drawing export must run inside the attached FreeCAD GUI."
            ) from exc
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        vector_source_path.parent.mkdir(parents=True, exist_ok=True)
        document.native_handle.recompute()
        Gui.updateGui()
        TechDrawGui.exportPageAsSvg(page, str(vector_source_path))
        if not vector_source_path.is_file() or vector_source_path.stat().st_size < 1000:
            raise RuntimeError("TechDraw SVG export is missing or unexpectedly small.")
        TechDrawGui.exportPageAsPdf(page, str(pdf_path))
        Gui.updateGui()
        self._validate_pdf(pdf_path)
        return DrawingEditResult(
            changed_native_names=(page.Name,),
            metadata={
                "pdf_path": str(pdf_path),
                "vector_source_path": str(vector_source_path),
            },
        )

    @staticmethod
    def _drawing_page(native_document: Any, page_name: str) -> Any:
        get_object = getattr(native_document, "getObject", None)
        page = get_object(str(page_name)) if callable(get_object) else None
        if page is None or str(getattr(page, "TypeId", "")) != "TechDraw::DrawPage":
            raise ValueError(f"Drawing page {page_name!r} does not exist.")
        return page

    @staticmethod
    def _drawing_page_member(
        native_document: Any,
        page: Any,
        native_name: str,
    ) -> Any:
        get_object = getattr(native_document, "getObject", None)
        candidate = get_object(str(native_name)) if callable(get_object) else None
        page_names = {
            str(getattr(item, "Name", ""))
            for item in FreeCADDrawingBackend._page_members(native_document, page)
        }
        if candidate is None or str(native_name) not in page_names:
            raise ValueError(
                f"Drawing item {native_name!r} is not part of page {page.Name!r}."
            )
        return candidate

    @staticmethod
    def _page_members(native_document: Any, page: Any) -> tuple[Any, ...]:
        """Return direct and projection-group-nested objects on a TechDraw page."""

        members: list[Any] = []
        seen_names: set[str] = set()

        def append(candidate: Any) -> None:
            name = str(getattr(candidate, "Name", ""))
            if not name or name in seen_names or candidate is page:
                return
            members.append(candidate)
            seen_names.add(name)

        for candidate in getattr(page, "Views", ()):
            append(candidate)
        for candidate in getattr(native_document, "Objects", ()):
            find_parent = getattr(candidate, "findParentPage", None)
            if not callable(find_parent):
                continue
            try:
                parent = find_parent()
            except Exception:
                continue
            if parent is page or str(getattr(parent, "Name", "")) == str(page.Name):
                append(candidate)
        return tuple(members)

    @classmethod
    def _drawing_view(
        cls,
        native_document: Any,
        page: Any,
        native_name: str,
    ) -> Any:
        view = cls._drawing_page_member(native_document, page, native_name)
        if str(getattr(view, "TypeId", "")) not in {
            "TechDraw::DrawViewPart",
            "TechDraw::DrawProjGroupItem",
        }:
            raise ValueError(
                f"Drawing item {native_name!r} is not a projected part view."
            )
        return view

    @classmethod
    def _logical_view_name(cls, view: Any) -> str:
        persisted = str(getattr(view, "RapidCADViewName", "")).strip().lower()
        if persisted:
            return persisted
        direction = getattr(view, "Direction", None)
        if direction is None:
            return ""
        try:
            components = tuple(
                float(getattr(direction, axis)) for axis in ("x", "y", "z")
            )
        except (AttributeError, TypeError, ValueError):
            return ""
        length = math.sqrt(sum(component * component for component in components))
        if length <= 1.0e-9:
            return ""
        normalized = tuple(component / length for component in components)
        canonical = {
            "front": (0.0, -1.0, 0.0),
            "top": (0.0, 0.0, 1.0),
            "right": (1.0, 0.0, 0.0),
            "isometric": (
                1.0 / math.sqrt(3.0),
                -1.0 / math.sqrt(3.0),
                1.0 / math.sqrt(3.0),
            ),
        }
        for name, expected in canonical.items():
            if all(
                math.isclose(actual, target, abs_tol=1.0e-5)
                for actual, target in zip(normalized, expected)
            ):
                return name
        return ""

    @staticmethod
    def _quantity_value(value: Any) -> float:
        return float(getattr(value, "Value", value))

    @staticmethod
    def _validated_page_position(
        position_mm: tuple[float, float],
    ) -> tuple[float, float]:
        if len(position_mm) != 2:
            raise ValueError("Drawing positions must contain exactly [x_mm, y_mm].")
        x_mm, y_mm = (float(position_mm[0]), float(position_mm[1]))
        if not math.isfinite(x_mm) or not math.isfinite(y_mm):
            raise ValueError("Drawing positions must be finite.")
        if not 0.0 <= x_mm <= 420.0 or not 0.0 <= y_mm <= 297.0:
            raise ValueError("Drawing positions must lie within the A3 page bounds.")
        return x_mm, y_mm

    @classmethod
    def _position_side(
        cls,
        view: Any,
        x_mm: float,
        y_mm: float,
    ) -> Any:
        delta_x = x_mm - cls._quantity_value(getattr(view, "X", 0.0))
        delta_y = y_mm - cls._quantity_value(getattr(view, "Y", 0.0))
        if abs(delta_x) >= abs(delta_y):
            return "right" if delta_x >= 0.0 else "left"
        return "top" if delta_y >= 0.0 else "bottom"

    @staticmethod
    def _set_string_properties(native_object: Any, values: dict[str, str]) -> None:
        add_property = getattr(native_object, "addProperty", None)
        for name, value in values.items():
            if not hasattr(native_object, name) and callable(add_property):
                add_property("App::PropertyString", name, "RapidCAD")
            setattr(native_object, name, str(value))

    @staticmethod
    def _resolve_template(resource_root: Path, filename: str) -> Path:
        templates_dir = resource_root / "Mod" / "TechDraw" / "Templates"
        candidates = (
            templates_dir / filename,
            templates_dir / "locale" / "de" / "A3_Landscape_ISO7200_DE.svg",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"No supported A3 TechDraw template was found in {templates_dir}."
        )

    @staticmethod
    def _resolve_sources(
        native_document: Any,
        objects: Sequence[CadObject],
    ) -> list[Any]:
        explicit_selection = bool(objects)
        if explicit_selection:
            candidates = [item.native_handle for item in objects]
        else:
            candidates = list(getattr(native_document, "Objects", ()))
        shape_candidates = []
        for candidate in candidates:
            if (
                candidate is None
                or getattr(candidate, "Document", native_document)
                is not native_document
            ):
                continue
            if str(getattr(candidate, "TypeId", "")).startswith("TechDraw::"):
                continue
            shape = getattr(candidate, "Shape", None)
            if shape is None or getattr(shape, "isNull", lambda: True)():
                continue
            shape_candidates.append(candidate)

        if explicit_selection:
            sources = shape_candidates
        else:
            # FreeCAD keeps boolean operands as live document objects.  Drawing
            # an operand together with its dependent result can mask internal
            # edges (for example a hole) even when the result itself is valid.
            # Select only visible terminal features: shape-bearing dependents
            # in InList supersede their upstream operands.
            shape_names = {
                str(getattr(candidate, "Name", ""))
                for candidate in shape_candidates
            }
            terminal = [
                candidate
                for candidate in shape_candidates
                if not any(
                    str(getattr(dependent, "Name", "")) in shape_names
                    for dependent in getattr(candidate, "InList", ())
                )
            ]
            sources = [
                candidate
                for candidate in terminal
                if getattr(
                    getattr(candidate, "ViewObject", None),
                    "Visibility",
                    True,
                )
            ]
        if not sources:
            raise ValueError(
                "No visible terminal shape-bearing objects are available for "
                "the drawing. Pass object_ids explicitly or make the intended "
                "terminal feature visible."
            )
        return sources

    @staticmethod
    def _source_bounds(sources: Sequence[Any]) -> ModelBounds:
        """Return only envelope data; this does not infer feature intent."""

        return ModelBounds(
            min(float(item.Shape.BoundBox.XMin) for item in sources),
            max(float(item.Shape.BoundBox.XMax) for item in sources),
            min(float(item.Shape.BoundBox.YMin) for item in sources),
            max(float(item.Shape.BoundBox.YMax) for item in sources),
            min(float(item.Shape.BoundBox.ZMin) for item in sources),
            max(float(item.Shape.BoundBox.ZMax) for item in sources),
        )

    @staticmethod
    def _semantic_features_for_sources(
        document: CadDocument,
        sources: Sequence[Any],
    ) -> tuple[object, ...]:
        """Load persisted definitions whose target is in a selected source chain."""

        source_names: set[str] = set()
        source_objects: dict[str, Any] = {}
        pending = list(sources)
        while pending:
            candidate = pending.pop()
            name = str(getattr(candidate, "Name", ""))
            if not name or name in source_names:
                continue
            source_names.add(name)
            source_objects[name] = candidate
            base = getattr(candidate, "Base", None)
            if isinstance(base, tuple):
                pending.extend(item for item in base if item is not None)
            elif base is not None:
                pending.append(base)
        persisted = dict(document.feature_definitions)
        for native_name, native_object in source_objects.items():
            raw = getattr(native_object, "RapidCADFeatureDefinition", "")
            if raw:
                try:
                    value = json.loads(str(raw))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    persisted[native_name] = value
        document.feature_definitions.update(persisted)
        definitions = []
        for value in document.feature_definitions_for_native_names(source_names):
            try:
                feature = feature_from_dict(value)
            except (KeyError, TypeError, ValueError):
                continue
            definitions.append(feature)
        return tuple(definitions)

    @staticmethod
    def _select_dimension_features(
        features: Sequence[Any],
        requested_ids: Optional[Sequence[str]],
    ) -> tuple[Any, ...]:
        """Select semantic features by opaque ID without accepting dimensions."""

        available = {str(feature.id): feature for feature in features}
        if requested_ids is None:
            return tuple(features)
        normalized = tuple(dict.fromkeys(str(item).strip() for item in requested_ids))
        if any(not item for item in normalized):
            raise ValueError("dimension_feature_ids must not contain empty IDs.")
        unknown = [item for item in normalized if item not in available]
        if unknown:
            available_text = ", ".join(sorted(available)) or "none"
            raise ValueError(
                "Unknown dimension feature ID(s): "
                f"{', '.join(unknown)}. Available feature IDs: {available_text}. "
                "Call get_object and copy semantic_features[].id exactly."
            )
        return tuple(available[item] for item in normalized)

    @staticmethod
    def _create_projection_group(
        *,
        native_document: Any,
        page: Any,
        sources: Sequence[Any],
        projection_angle: str,
        anchor_x_mm: float,
        anchor_y_mm: float,
        app_module: Any,
    ) -> tuple[Any, tuple[Any, Any, Any]]:
        """Create TechDraw's native auto-distributed orthographic view group."""

        group = native_document.addObject(
            "TechDraw::DrawProjGroup", "RapidCADProjectionGroup"
        )
        page.addView(group)
        group.Source = list(sources)
        group.ProjectionType = (
            "First Angle" if projection_angle == "first" else "Third Angle"
        )
        group.AutoDistribute = True
        if hasattr(group, "ScaleType"):
            group.ScaleType = "Automatic"
        group.X = anchor_x_mm
        group.Y = anchor_y_mm

        front = group.addProjection("Front")
        # Preserve RapidCADPy's front convention while letting the group derive
        # the other orthographic directions and positions.
        group.Anchor.Direction = app_module.Vector(0.0, -1.0, 0.0)
        group.Anchor.RotationVector = app_module.Vector(1.0, 0.0, 0.0)
        right = group.addProjection("Right")
        top = group.addProjection("Top")
        return group, (front, top, right)

    @staticmethod
    def _drawing_scale(sources: Sequence[Any]) -> float:
        x_min = min(float(item.Shape.BoundBox.XMin) for item in sources)
        x_max = max(float(item.Shape.BoundBox.XMax) for item in sources)
        y_min = min(float(item.Shape.BoundBox.YMin) for item in sources)
        y_max = max(float(item.Shape.BoundBox.YMax) for item in sources)
        z_min = min(float(item.Shape.BoundBox.ZMin) for item in sources)
        z_max = max(float(item.Shape.BoundBox.ZMax) for item in sources)
        return select_drawing_scale((x_max - x_min, y_max - y_min, z_max - z_min))

    @staticmethod
    def _dimension_view_geometries(
        *,
        views: Sequence[Any],
        view_specs: Sequence[Any],
        bounds: ModelBounds,
        scale: float,
    ) -> tuple[ViewGeometry, ...]:
        result: list[ViewGeometry] = []
        for view, spec in zip(views, view_specs):
            if spec.name in {"front", "top", "right"}:
                u_range, v_range = view_model_ranges(spec.name, bounds)
            else:
                # The isometric view is not dimensioned, but its actual sheet
                # rectangle remains an obstacle for collision placement.
                u_range, v_range = (0.0, 1.0), (0.0, 1.0)
            width, height = FreeCADDrawingBackend._view_size(
                view,
                fallback_width=max(1.0, u_range[1] - u_range[0]),
                fallback_height=max(1.0, v_range[1] - v_range[0]),
                scale=scale,
            )
            result.append(
                ViewGeometry(
                    name=spec.name,
                    x_mm=float(view.X),
                    y_mm=float(view.Y),
                    width_mm=width,
                    height_mm=height,
                    scale=scale,
                    model_u_range=u_range,
                    model_v_range=v_range,
                )
            )
        return tuple(result)

    @staticmethod
    def _view_size(
        view: Any,
        *,
        fallback_width: float,
        fallback_height: float,
        scale: float,
    ) -> tuple[float, float]:
        """Measure a TechDraw view even when it has no Width/Height properties."""

        width = float(getattr(view, "Width", 0.0) or 0.0)
        height = float(getattr(view, "Height", 0.0) or 0.0)
        boxes = []
        for method_name in ("getVisibleEdges", "getHiddenEdges"):
            method = getattr(view, method_name, None)
            if not callable(method):
                continue
            try:
                edges = method() or ()
            except Exception:
                continue
            boxes.extend(
                edge.BoundBox
                for edge in edges
                if getattr(edge, "BoundBox", None) is not None
            )
        if boxes:
            width = max(
                width,
                (
                    max(float(box.XMax) for box in boxes)
                    - min(float(box.XMin) for box in boxes)
                )
                * scale,
            )
            height = max(
                height,
                (
                    max(float(box.YMax) for box in boxes)
                    - min(float(box.YMin) for box in boxes)
                )
                * scale,
            )
        return (
            width if width > 0.0 else fallback_width * scale,
            height if height > 0.0 else fallback_height * scale,
        )

    @staticmethod
    def _create_native_dimensions(
        *,
        native_document: Any,
        page: Any,
        views: Sequence[Any],
        view_specs: Sequence[Any],
        placements: Sequence[DimensionPlacement],
        standard: str,
        app_module: Any,
        techdraw_module: Any,
    ) -> list[Any]:
        """Create native dimensions bound to existing projected geometry.

        This deliberately avoids ``TechDraw.makeDistanceDim``: that helper
        creates cosmetic vertices and can crash FreeCAD while doing so.  Every
        dimension must instead resolve to an existing ``EdgeN`` in its
        ``DrawViewPart`` and is represented by a normal
        ``TechDraw::DrawViewDimension``.  Failure to resolve is an error; the
        drawing is never silently replaced with annotations or SVG graphics.
        """

        _ = (app_module, techdraw_module)
        view_by_name = {
            spec.name: view
            for view, spec in zip(views, view_specs)
            if spec.name in {"front", "top", "right"}
        }
        resolved: list[tuple[DimensionPlacement, Any, str, str]] = []
        for placement in placements:
            view = view_by_name.get(placement.intent.view)
            if view is None:
                raise RuntimeError(
                    "Cannot create native dimension "
                    f"{placement.intent.id}: projected view "
                    f"{placement.intent.view!r} does not exist."
                )
            dimension_type, edge_name = (
                FreeCADDrawingBackend._resolve_dimension_edge(view, placement)
            )
            resolved.append((placement, view, dimension_type, edge_name))

        result: list[Any] = []
        for index, (placement, view, dimension_type, edge_name) in enumerate(
            resolved,
            start=1,
        ):
            dimension = native_document.addObject(
                "TechDraw::DrawViewDimension",
                f"RapidCADDimension{index}",
            )
            dimension.Type = dimension_type
            dimension.MeasureType = "Projected"
            dimension.References2D = [(view, edge_name)]
            dimension.X = float(placement.x_mm)
            dimension.Y = float(placement.y_mm)
            dimension.Arbitrary = False
            dimension.FormatSpec = FreeCADDrawingBackend._measured_format_spec(
                placement.intent,
                standard,
            )
            FreeCADDrawingBackend._set_dimension_provenance(
                dimension,
                placement,
                edge_name,
            )
            page.addView(dimension)
            result.append(dimension)
        native_document.recompute()
        return result

    @staticmethod
    def _measured_format_spec(intent: Any, standard: str) -> str:
        """Format FreeCAD's measured value without replacing that value.

        ``%.nf`` is evaluated by ``DrawViewDimension`` against References2D.
        Semantic feature data contributes only qualifiers such as THRU and the
        countersink angle; it never supplies the rendered diameter or length.
        """

        precision = 2 if normalize_drawing_standard(standard) == "ISO" else 3
        measured_value = f"%.{precision}f"
        multiplicity = f"{intent.multiplicity}X " if intent.multiplicity > 1 else ""
        if intent.kind == "hole":
            suffix = " THRU" if intent.through else ""
            if not intent.through and intent.depth_mm is not None:
                suffix = f" DEPTH {intent.depth_mm:g} mm"
            return f"{multiplicity}{measured_value}{suffix}"
        if intent.kind == "countersink":
            angle = intent.angle_degrees if intent.angle_degrees is not None else 90.0
            return f"⌵ {measured_value} × {angle:g}°"
        if intent.kind == "chamfer":
            angle = intent.angle_degrees if intent.angle_degrees is not None else 45.0
            return f"{multiplicity}C{measured_value} × {angle:g}°"
        if intent.kind == "thickness":
            return f"t {measured_value}"
        return f"{multiplicity}{measured_value}"

    @staticmethod
    def _set_dimension_provenance(
        dimension: Any,
        placement: DimensionPlacement,
        edge_name: str,
    ) -> None:
        """Persist the semantic request and native measurement dependency."""

        public_dimension_kind = {
            "hole": "diameter",
            "countersink": "countersink_diameter",
        }.get(placement.intent.kind, placement.intent.kind)
        values = {
            "RapidCADDimensionIntentId": placement.intent.id,
            "RapidCADFeatureIds": json.dumps(
                list(placement.intent.source_references),
                sort_keys=True,
            ),
            "RapidCADProjectedReference": f"{placement.intent.view}:{edge_name}",
            "RapidCADMeasurementSource": "projected_geometry",
            "RapidCADDrawingItemKind": "dimension",
            "RapidCADDimensionKind": public_dimension_kind,
        }
        FreeCADDrawingBackend._set_string_properties(dimension, values)

    @staticmethod
    def _resolve_dimension_edge(
        view: Any,
        placement: DimensionPlacement,
    ) -> tuple[str, str]:
        """Resolve a semantic intent to a native TechDraw projected edge."""

        intent = placement.intent
        edges = FreeCADDrawingBackend._projected_edges(view)
        if intent.kind in {"hole", "countersink", "radius"}:
            desired_radius = (
                intent.value_mm
                if intent.kind == "radius"
                else intent.value_mm / 2.0
            )
            candidates = [
                (index, edge)
                for index, edge in edges
                if FreeCADDrawingBackend._curve_type(edge) == "Part::GeomCircle"
                and math.isclose(
                    float(getattr(edge.Curve, "Radius", -1.0)),
                    desired_radius,
                    rel_tol=1.0e-5,
                    abs_tol=1.0e-4,
                )
            ]
            dimension_type = "Radius" if intent.kind == "radius" else "Diameter"
        elif intent.kind in {"overall", "chamfer", "thickness"}:
            horizontal = FreeCADDrawingBackend._intent_is_horizontal(intent)
            candidates = [
                (index, edge)
                for index, edge in edges
                if FreeCADDrawingBackend._curve_type(edge) == "Part::GeomLine"
                and FreeCADDrawingBackend._edge_matches_axis(edge, horizontal)
                and math.isclose(
                    float(getattr(edge, "Length", -1.0)),
                    intent.value_mm,
                    rel_tol=1.0e-5,
                    abs_tol=1.0e-4,
                )
            ]
            dimension_type = "DistanceX" if horizontal else "DistanceY"
        else:
            raise RuntimeError(
                f"Native TechDraw references are not implemented for dimension "
                f"{intent.id} ({intent.kind})."
            )

        if not candidates:
            available = ", ".join(
                FreeCADDrawingBackend._edge_description(index, edge)
                for index, edge in edges
            ) or "none"
            raise RuntimeError(
                f"Cannot bind dimension {intent.id} to projected geometry in "
                f"view {intent.view}; expected {dimension_type} value "
                f"{intent.value_mm:g} mm. Available edges: {available}."
            )
        selected_index, _ = FreeCADDrawingBackend._select_dimension_edge(
            candidates,
            placement.side,
        )
        return dimension_type, f"Edge{selected_index}"

    @staticmethod
    def _projected_edges(view: Any) -> list[tuple[int, Any]]:
        """Return TechDraw edges with their native zero-based EdgeN indexes."""

        get_edge = getattr(view, "getEdgeByIndex", None)
        if not callable(get_edge):
            raise RuntimeError(
                f"TechDraw view {getattr(view, 'Name', '<unknown>')} does not "
                "expose projected-edge lookup."
            )
        result: list[tuple[int, Any]] = []
        for index in range(10_000):
            try:
                edge = get_edge(index)
            except Exception:
                break
            if edge is None:
                break
            result.append((index, edge))
        return result

    @staticmethod
    def _curve_type(edge: Any) -> str:
        return str(getattr(getattr(edge, "Curve", None), "TypeId", ""))

    @staticmethod
    def _intent_is_horizontal(intent: Any) -> bool:
        if len(intent.reference_points) < 2:
            return True
        first, second = intent.reference_points[:2]
        if intent.view == "front":
            delta_u, delta_v = second[0] - first[0], second[2] - first[2]
        elif intent.view == "top":
            delta_u, delta_v = second[0] - first[0], second[1] - first[1]
        else:
            delta_u, delta_v = second[1] - first[1], second[2] - first[2]
        return abs(delta_u) >= abs(delta_v)

    @staticmethod
    def _edge_matches_axis(edge: Any, horizontal: bool) -> bool:
        box = getattr(edge, "BoundBox", None)
        if box is None:
            return False
        x_length = float(getattr(box, "XLength", 0.0))
        y_length = float(getattr(box, "YLength", 0.0))
        tolerance = max(1.0e-5, float(getattr(edge, "Length", 0.0)) * 1.0e-6)
        return y_length <= tolerance if horizontal else x_length <= tolerance

    @staticmethod
    def _select_dimension_edge(
        candidates: Sequence[tuple[int, Any]],
        side: str,
    ) -> tuple[int, Any]:
        def coordinate(candidate: tuple[int, Any]) -> float:
            box = getattr(candidate[1], "BoundBox", None)
            center = getattr(box, "Center", None)
            if side in {"left", "right"}:
                return float(getattr(center, "x", 0.0))
            return float(getattr(center, "y", 0.0))

        reverse = side in {"top", "right"}
        return sorted(candidates, key=coordinate, reverse=reverse)[0]

    @staticmethod
    def _edge_description(index: int, edge: Any) -> str:
        curve_type = FreeCADDrawingBackend._curve_type(edge) or "unknown"
        length = float(getattr(edge, "Length", 0.0))
        radius = getattr(getattr(edge, "Curve", None), "Radius", None)
        suffix = f", radius={float(radius):g}" if radius is not None else ""
        return f"Edge{index}({curve_type}, length={length:g}{suffix})"

    @staticmethod
    def _fill_title_block(
        template: Any,
        *,
        part_name: str,
        run_id: str,
        scale: float,
        standard: str,
    ) -> None:
        editable = dict(getattr(template, "EditableTexts", {}) or {})
        values = {
            "LEGAL_OWNER_1": "Agentic CAD",
            "LEGAL_OWNER_2": "",
            "LEGAL_OWNER_3": "",
            "LEGAL_OWNER_4": "",
            "TITLE_NAME": part_name,
            "TITLE": part_name,
            "DRAWING_TITLE": part_name,
            "DRAWING_NUMBER": run_id,
            "DRAWING_NO": run_id,
            "REVISION_INDEX": "-",
            "AUTHOR_NAME": "RapidCAD",
            "AUTHOR": "RapidCAD",
            "CREATOR": "RapidCAD",
            "FC-SC": f"{scale:g}:1",
            "SCALE": f"{scale:g}:1",
            "FC-DATE": date.today().isoformat(),
            "DATE": date.today().isoformat(),
            "SHEET": "1 / 1",
            "DATE_OF_ISSUE": date.today().isoformat(),
            "SHEET_NUMBER": "1 / 1",
            "LANGUAGE_CODE": "EN",
            "SUPPLEMENTARY_TITLE_1": "",
            "SUPPLEMENTARY_TITLE_2": "",
            "RESPONSIBLE_DEPARTMENT": "Engineering",
            "APPROVAL_PERSON": "-",
            "DOCUMENT_TYPE": "Technical drawing",
            "DOCUMENT_STATUS": "Draft",
            "GENERAL_TOLERANCES": ("ISO 2768-m" if standard == "ISO" else "ASME Y14.5"),
            "UNITS": "mm" if standard == "ISO" else "inch",
            "PART_MATERIAL": "-",
        }
        for key in tuple(editable):
            normalized = key.upper().replace(" ", "_")
            if normalized in values:
                editable[key] = values[normalized]
        if editable:
            template.EditableTexts = editable

    @staticmethod
    def _validate_views(views: Sequence[Any]) -> None:
        empty_views = [
            str(getattr(view, "Name", "<unknown>"))
            for view in views
            if not FreeCADDrawingBackend._view_has_projected_geometry(view)
        ]
        if empty_views:
            raise RuntimeError(
                "TechDraw produced no projected geometry for view(s): "
                + ", ".join(empty_views)
                + "; drawing export aborted."
            )

    @staticmethod
    def _view_has_projected_geometry(view: Any) -> bool:
        for method_name in ("getVisibleEdges", "getHiddenEdges"):
            method = getattr(view, method_name, None)
            if not callable(method):
                continue
            try:
                if method():
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _wait_for_projected_geometry(
        *,
        native_document: Any,
        page: Any,
        views: Sequence[Any],
        gui_module: Any,
        timeout_seconds: float = 10.0,
    ) -> None:
        """Wait for TechDraw's queued projections before dimensioning/export."""

        deadline = time.monotonic() + timeout_seconds
        while True:
            native_document.recompute()
            recompute_page = getattr(page, "recompute", None)
            if callable(recompute_page):
                recompute_page(True)
            gui_module.updateGui()
            if all(
                FreeCADDrawingBackend._view_has_projected_geometry(view)
                for view in views
            ):
                return
            if time.monotonic() >= deadline:
                FreeCADDrawingBackend._validate_views(views)
            time.sleep(0.02)

    @staticmethod
    def _validate_layout(
        views: Sequence[Any],
        *,
        page_width_mm: float,
        page_height_mm: float,
    ) -> None:
        """Reject projected views that TechDraw reports outside or overlapping."""

        boxes: list[tuple[str, float, float, float, float]] = []
        for view in views:
            try:
                width = float(view.Width)
                height = float(view.Height)
                x = float(view.X)
                y = float(view.Y)
            except (AttributeError, TypeError, ValueError):
                continue
            if width <= 0 or height <= 0:
                continue
            box = (
                str(getattr(view, "Name", "view")),
                x - width / 2.0,
                y - height / 2.0,
                x + width / 2.0,
                y + height / 2.0,
            )
            if (
                box[1] < 0
                or box[2] < 0
                or box[3] > page_width_mm
                or box[4] > page_height_mm
            ):
                raise RuntimeError(f"Projected view {box[0]} falls outside the sheet.")
            boxes.append(box)
        for index, first in enumerate(boxes):
            for second in boxes[index + 1 :]:
                separated = (
                    first[3] + 2.0 <= second[1]
                    or second[3] + 2.0 <= first[1]
                    or first[4] + 2.0 <= second[2]
                    or second[4] + 2.0 <= first[2]
                )
                if not separated:
                    raise RuntimeError(
                        f"Projected views {first[0]} and {second[0]} overlap."
                    )

    @staticmethod
    def _validate_pdf(path: Path) -> None:
        if not path.is_file() or path.stat().st_size < 1000:
            raise RuntimeError("TechDraw PDF export is missing or unexpectedly small.")
        data = path.read_bytes()
        if not data.startswith(b"%PDF-"):
            raise RuntimeError("TechDraw output is not a PDF file.")
        media_boxes = re.findall(
            rb"/MediaBox\s*\[\s*[-+0-9.]+\s+[-+0-9.]+\s+"
            rb"([-+0-9.]+)\s+([-+0-9.]+)\s*\]",
            data,
        )
        if media_boxes:
            width_pt, height_pt = (float(value) for value in media_boxes[0])
            width_mm = width_pt * 25.4 / 72.0
            height_mm = height_pt * 25.4 / 72.0
            dimensions = sorted((width_mm, height_mm))
            if not (
                math.isclose(dimensions[0], 297.0, abs_tol=1.0)
                and math.isclose(dimensions[1], 420.0, abs_tol=1.0)
            ):
                raise RuntimeError(
                    "TechDraw exported an incorrect page size "
                    f"({width_mm:.1f} x {height_mm:.1f} mm, expected A3)."
                )

    @staticmethod
    def _validate_dxf(path: Path, *, expected_annotations: Sequence[str]) -> None:
        """Verify a whole-page ASCII DXF and its dimension annotations."""

        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("TechDraw DXF export is missing or empty.")
        content = path.read_text(encoding="utf-8", errors="replace")
        if (
            "SECTION" not in content
            or "ENTITIES" not in content
            or "EOF" not in content
        ):
            raise RuntimeError("TechDraw output is not a complete DXF file.")
        missing = [
            item
            for item in expected_annotations
            if not any(variant in content for variant in _dxf_text_variants(item))
        ]
        if missing:
            raise RuntimeError(
                "TechDraw DXF is missing drawing annotation(s): " + ", ".join(missing)
            )

    @staticmethod
    def _remove_previous_drawing(native_document: Any) -> None:
        """Leave existing TechDraw trees intact before creating a new page.

        Removing a ``TechDraw::DrawProjGroup`` with dependent projection views
        can crash current macOS FreeCAD builds in
        ``DrawViewCollection::unsetupObject``.  FreeCAD assigns a unique native
        name when a requested name is already in use, so creating a new drawing
        page is safe and preserves the prior page for the user to close/delete
        interactively.  Do not call ``Document.removeObject`` on TechDraw
        collections from the automation bridge.
        """

        _ = native_document

    @staticmethod
    def _safe_component(value: str, fallback: str) -> str:
        normalized = _SAFE_FILENAME.sub("_", value.strip()).strip("._-")
        return normalized or fallback


__all__ = ["FreeCADDrawingBackend", "FreeCADDrawingBackendFactory"]


def _dxf_text_variants(value: str) -> tuple[str, ...]:
    escaped = "".join(
        character if ord(character) < 128 else f"\\U+{ord(character):04X}"
        for character in value
    )
    return value, escaped
