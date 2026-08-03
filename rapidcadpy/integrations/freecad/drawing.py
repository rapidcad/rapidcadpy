"""FreeCAD TechDraw implementation of RapidCADPy technical drawings."""

from __future__ import annotations

import math
import re
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence

from ...cad_objects import CadDocument, CadObject
from ...drawing import (
    DrawingBackend,
    DrawingBackendFactory,
    DrawingResult,
    normalize_projection_angle,
    projected_view_layout,
    select_drawing_scale,
)

_TEMPLATES = {
    ("ISO", "A3", None): "A3_Landscape_ISO5457_advanced.svg",
    ("ISO", "A3", "iso-a3-v1"): "A3_Landscape_ISO5457_advanced.svg",
}
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_DRAWING_NAMES = (
    "RapidCADDrawing",
    "RapidCADTemplate",
    "RapidCADFront",
    "RapidCADTop",
    "RapidCADRight",
    "RapidCADIsometric",
    "RapidCADDimensions",
)


class FreeCADDrawingBackendFactory(DrawingBackendFactory):
    """Construct FreeCAD drawing backends without leaking native handles."""

    def create(self, document: CadDocument) -> DrawingBackend:
        if document.backend.strip().lower() != "freecad":
            raise ValueError("FreeCADDrawingBackend requires a FreeCAD document.")
        return FreeCADDrawingBackend()


class FreeCADDrawingBackend(DrawingBackend):
    """Create an A3 ISO TechDraw page from live native FreeCAD objects."""

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
    ) -> DrawingResult:
        normalized_standard = standard.strip().upper()
        normalized_sheet = sheet_size.strip().upper()
        normalized_projection = normalize_projection_angle(projection_angle)
        normalized_template = template_id.strip().lower() if template_id else None
        template_filename = _TEMPLATES.get(
            (normalized_standard, normalized_sheet, normalized_template)
        )
        if template_filename is None:
            raise ValueError(
                "FreeCAD drawing export currently supports ISO A3 with template "
                "'iso-a3-v1' (or no explicit template_id)."
            )

        try:
            import FreeCAD as App
            import FreeCADGui as Gui
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
        template = native_document.addObject(
            "TechDraw::DrawSVGTemplate", "RapidCADTemplate"
        )
        template.Template = str(template_path)
        page.Template = template
        native_document.recompute()

        scale = self._drawing_scale(sources)
        created_names = [page.Name, template.Name]
        view_specs = projected_view_layout(
            normalized_projection,
            sheet_size=normalized_sheet,
        )
        views = []
        for spec in view_specs:
            native_name = f"RapidCAD{spec.name.title()}"
            view = native_document.addObject("TechDraw::DrawViewPart", native_name)
            view.Source = list(sources)
            view.Direction = App.Vector(*spec.direction)
            if hasattr(view, "ScaleType"):
                view.ScaleType = "Custom"
            view.Scale = scale
            page.addView(view)
            view.X = spec.x_mm
            view.Y = spec.y_mm
            views.append(view)
            created_names.append(view.Name)

        dimensions = native_document.addObject(
            "TechDraw::DrawViewAnnotation", "RapidCADDimensions"
        )
        dimensions.Text = self._overall_dimension_lines(sources)
        if hasattr(dimensions, "TextSize"):
            dimensions.TextSize = 3.5
        page.addView(dimensions)
        dimensions.X = 300.0
        dimensions.Y = 65.0
        created_names.append(dimensions.Name)

        self._fill_title_block(
            template,
            part_name=part_name,
            run_id=run_id,
            scale=scale,
        )
        if hasattr(page, "KeepUpdated"):
            page.KeepUpdated = True
        page.ViewObject.show()
        native_document.recompute()
        Gui.updateGui()
        native_document.recompute()

        self._validate_views(views)
        self._validate_layout(
            [*views, dimensions],
            page_width_mm=420.0,
            page_height_mm=297.0,
        )
        safe_part = self._safe_component(part_name, "drawing")
        safe_run = self._safe_component(run_id, "run")
        pdf_path = output_directory / (
            f"{safe_part}_{safe_run}_{date.today().isoformat()}.pdf"
        )
        vector_directory = output_directory / ".intermediate"
        vector_directory.mkdir(parents=True, exist_ok=True)
        vector_source_path = vector_directory / (
            f".{safe_part}_{safe_run}_{date.today().isoformat()}.svg"
        )
        TechDrawGui.exportPageAsSvg(page, str(vector_source_path))
        if not vector_source_path.is_file() or vector_source_path.stat().st_size < 1000:
            raise RuntimeError("TechDraw SVG export is missing or unexpectedly small.")
        TechDrawGui.exportPageAsPdf(page, str(pdf_path))
        Gui.updateGui()
        self._validate_pdf(pdf_path)

        native_path: Optional[Path] = None
        warnings = [
            "Overall envelope dimensions are TechDraw annotations; "
            "feature-level associative dimensions are not generated yet."
        ]
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
                "template_id": normalized_template or "iso-a3-v1",
                "projection_angle": normalized_projection,
                "scale": scale,
                "views": [spec.to_dict() for spec in view_specs],
                "vector_export": True,
                "minimum_text_height_mm": 3.5,
                "parametric_link_preserved": True,
            },
        )

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
        sources = []
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
            visible = getattr(
                getattr(candidate, "ViewObject", None), "Visibility", True
            )
            if explicit_selection or visible:
                sources.append(candidate)
        if not sources:
            raise ValueError(
                "No visible shape-bearing objects are available for the drawing."
            )
        return sources

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
    def _overall_dimension_lines(sources: Sequence[Any]) -> list[str]:
        x_min = min(float(item.Shape.BoundBox.XMin) for item in sources)
        x_max = max(float(item.Shape.BoundBox.XMax) for item in sources)
        y_min = min(float(item.Shape.BoundBox.YMin) for item in sources)
        y_max = max(float(item.Shape.BoundBox.YMax) for item in sources)
        z_min = min(float(item.Shape.BoundBox.ZMin) for item in sources)
        z_max = max(float(item.Shape.BoundBox.ZMax) for item in sources)
        return [
            "OVERALL DIMENSIONS",
            f"X  {x_max - x_min:.2f} mm",
            f"Y  {y_max - y_min:.2f} mm",
            f"Z  {z_max - z_min:.2f} mm",
        ]

    @staticmethod
    def _fill_title_block(
        template: Any,
        *,
        part_name: str,
        run_id: str,
        scale: float,
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
            "GENERAL_TOLERANCES": "ISO 2768-m",
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
        visible_geometry = False
        for view in views:
            for method_name in ("getVisibleEdges", "getHiddenEdges"):
                method = getattr(view, method_name, None)
                if callable(method):
                    try:
                        if method():
                            visible_geometry = True
                    except Exception:
                        pass
        if not visible_geometry:
            raise RuntimeError(
                "TechDraw produced no visible projected geometry; PDF export aborted."
            )

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
    def _remove_previous_drawing(native_document: Any) -> None:
        for name in reversed(_DRAWING_NAMES):
            existing = native_document.getObject(name)
            if existing is not None:
                native_document.removeObject(existing.Name)
        existing_source = native_document.getObject("RapidCADDrawingSource")
        if existing_source is not None:
            native_document.removeObject(existing_source.Name)
        native_document.recompute()

    @staticmethod
    def _safe_component(value: str, fallback: str) -> str:
        normalized = _SAFE_FILENAME.sub("_", value.strip()).strip("._-")
        return normalized or fallback


__all__ = ["FreeCADDrawingBackend", "FreeCADDrawingBackendFactory"]
