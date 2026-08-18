"""Backend-neutral technical drawing contracts and backend factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Dict, Literal, Optional, Sequence, Type

from .cad_objects import CadDocument, CadObject


ProjectionAngle = Literal["first", "third"]
DrawingStandard = Literal["ISO", "ASME"]
DimensionKind = Literal[
    "overall",
    "hole",
    "countersink",
    "spacing",
    "radius",
    "chamfer",
    "thickness",
]
DimensionView = Literal["front", "top", "right"]
DimensionSide = Literal["top", "bottom", "left", "right"]
DrawingOutputFormat = Literal["pdf", "idw", "dwg", "dxf"]
DRAWING_OUTPUT_FORMATS: tuple[DrawingOutputFormat, ...] = (
    "pdf",
    "idw",
    "dwg",
    "dxf",
)


def normalize_drawing_standard(value: str) -> DrawingStandard:
    """Return the supported engineering-drawing standard name."""

    normalized = str(value).strip().upper().replace("-", "")
    aliases: dict[str, DrawingStandard] = {
        "ISO": "ISO",
        "ASME": "ASME",
        "ANSI": "ASME",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError("standard must be 'ISO' or 'ASME'.") from exc


def format_dimension_value(value_mm: float, standard: str) -> str:
    """Format a millimetre value using the requested drawing convention."""

    normalized = normalize_drawing_standard(standard)
    if not math.isfinite(value_mm) or value_mm < 0:
        raise ValueError("Dimension values must be finite and non-negative.")
    if normalized == "ISO":
        return f"{value_mm:.2f}".replace(".", ",")
    return f"{value_mm / 25.4:.3f}"


@dataclass(frozen=True)
class DimensionIntent:
    """Backend-neutral semantic dimension before sheet placement.

    Coordinates remain in model millimetres. Adapters project them into the
    requested orthographic view and create native dimension objects.
    """

    id: str
    kind: DimensionKind
    view: DimensionView
    value_mm: float
    reference_points: tuple[tuple[float, float, float], ...]
    preferred_sides: tuple[DimensionSide, ...]
    depth_mm: Optional[float] = None
    angle_degrees: Optional[float] = None
    through: bool = False
    multiplicity: int = 1
    source_references: tuple[str, ...] = ()

    def formatted_text(self, standard: str) -> str:
        """Return an explicit ISO/ASME label for TechDraw's format override."""

        value = format_dimension_value(self.value_mm, standard)
        prefix = f"{self.multiplicity}X " if self.multiplicity > 1 else ""
        if self.kind == "hole":
            text = f"{prefix}\u2300{value}"
            if self.through:
                return f"{text} THRU"
            if self.depth_mm is not None:
                return f"{text} \u21a7{format_dimension_value(self.depth_mm, standard)}"
            return text
        if self.kind == "countersink":
            angle = self.angle_degrees if self.angle_degrees is not None else 90.0
            return f"⌵ ⌀{value} × {angle:g}°"
        if self.kind == "radius":
            return f"{prefix}R{value}"
        if self.kind == "chamfer":
            angle = self.angle_degrees if self.angle_degrees is not None else 45.0
            return f"{prefix}C{value} \u00d7 {angle:g}\u00b0"
        if self.kind == "thickness":
            return f"t {value}"
        return value

    def to_dict(self, standard: str) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "view": self.view,
            "value_mm": self.value_mm,
            "depth_mm": self.depth_mm,
            "angle_degrees": self.angle_degrees,
            "through": self.through,
            "multiplicity": self.multiplicity,
            "reference_points": [list(point) for point in self.reference_points],
            "preferred_sides": list(self.preferred_sides),
            "source_references": list(self.source_references),
            "formatted_text": self.formatted_text(standard),
        }


@dataclass(frozen=True)
class DimensionPlacement:
    """Collision-checked sheet placement for one semantic dimension."""

    intent: DimensionIntent
    side: DimensionSide
    x_mm: float
    y_mm: float
    text_box: tuple[float, float, float, float]
    projected_reference_points: tuple[tuple[float, float], ...]
    leader_segments: tuple[tuple[float, float, float, float], ...] = ()

    def to_dict(self, standard: str) -> dict[str, object]:
        result = self.intent.to_dict(standard)
        result.update(
            {
                "side": self.side,
                "x_mm": self.x_mm,
                "y_mm": self.y_mm,
                "text_box": list(self.text_box),
                "projected_reference_points": [
                    list(point) for point in self.projected_reference_points
                ],
                "leader_segments": [list(segment) for segment in self.leader_segments],
            }
        )
        return result


def normalize_output_formats(
    values: Optional[Sequence[str]],
) -> tuple[DrawingOutputFormat, ...]:
    """Validate and de-duplicate backend-neutral drawing output formats."""

    requested = values if values is not None else ("pdf",)
    normalized: list[DrawingOutputFormat] = []
    for value in requested:
        candidate = str(value).strip().lower()
        if candidate not in DRAWING_OUTPUT_FORMATS:
            supported = ", ".join(DRAWING_OUTPUT_FORMATS)
            raise ValueError(
                f"Unsupported drawing output format {value!r}; expected one of: "
                f"{supported}."
            )
        typed_candidate: DrawingOutputFormat = candidate  # type: ignore[assignment]
        if typed_candidate not in normalized:
            normalized.append(typed_candidate)
    if not normalized:
        raise ValueError("output_formats must contain at least one format.")
    return tuple(normalized)


@dataclass(frozen=True)
class DrawingViewSpec:
    """Backend-neutral placement and camera direction for one drawing view."""

    name: str
    direction: tuple[float, float, float]
    x_mm: float
    y_mm: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "direction": list(self.direction),
            "x_mm": self.x_mm,
            "y_mm": self.y_mm,
        }


def normalize_projection_angle(value: str) -> ProjectionAngle:
    """Validate a public projection-angle value without backend terminology."""

    normalized = value.strip().lower().replace("_", "-")
    aliases: dict[str, ProjectionAngle] = {
        "first": "first",
        "first-angle": "first",
        "third": "third",
        "third-angle": "third",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError("projection_angle must be 'first' or 'third'.") from exc


def projected_view_layout(
    projection_angle: str,
    *,
    sheet_size: str = "A3",
) -> tuple[DrawingViewSpec, ...]:
    """Return the standard four-view layout for a drawing sheet.

    First-angle projection places the top view below the front view and the
    right-side view to its left. Third-angle projection reverses those two
    placements. Coordinates are expressed in sheet millimetres so adapters do
    not leak native coordinate objects through the public contract.
    """

    if sheet_size.strip().upper() != "A3":
        raise ValueError("Projected-view layout currently supports A3 sheets only.")
    angle = normalize_projection_angle(projection_angle)
    if angle == "first":
        positions = {
            "front": (170.0, 170.0),
            "top": (170.0, 65.0),
            "right": (60.0, 170.0),
            "isometric": (315.0, 175.0),
        }
    else:
        positions = {
            "front": (105.0, 120.0),
            "top": (105.0, 225.0),
            "right": (215.0, 120.0),
            "isometric": (315.0, 185.0),
        }
    directions = {
        "front": (0.0, -1.0, 0.0),
        "top": (0.0, 0.0, 1.0),
        "right": (1.0, 0.0, 0.0),
        "isometric": (1.0, -1.0, 1.0),
    }
    return tuple(
        DrawingViewSpec(
            name=name,
            direction=directions[name],
            x_mm=positions[name][0],
            y_mm=positions[name][1],
        )
        for name in ("front", "top", "right", "isometric")
    )


def select_drawing_scale(extents_mm: Sequence[float]) -> float:
    """Select the largest RAP-50 engineering scale that fits an A3 layout."""

    if len(extents_mm) != 3 or any(value <= 0 for value in extents_mm):
        raise ValueError("extents_mm must contain three positive dimensions.")
    largest = max(float(value) for value in extents_mm)
    for scale in (2.0, 1.0, 0.5, 0.2):
        if largest * scale <= 100.0:
            return scale
    raise ValueError(
        "The selected geometry does not fit the A3 four-view layout at 1:5 scale."
    )


@dataclass(frozen=True)
class DrawingResult:
    """Files and native objects produced by one synchronous drawing export."""

    pdf_path: Optional[Path] = None
    output_paths: dict[str, Path] = field(default_factory=dict)
    vector_source_path: Optional[Path] = None
    native_drawing_path: Optional[Path] = None
    page_name: Optional[str] = None
    source_object_ids: tuple[str, ...] = ()
    created_native_names: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        output_paths = {key: str(value) for key, value in self.output_paths.items()}
        if self.pdf_path is not None:
            output_paths.setdefault("pdf", str(self.pdf_path))
        return {
            "pdf_path": str(self.pdf_path) if self.pdf_path is not None else None,
            "output_paths": output_paths,
            "vector_source_path": (
                str(self.vector_source_path)
                if self.vector_source_path is not None
                else None
            ),
            "native_drawing_path": (
                str(self.native_drawing_path)
                if self.native_drawing_path is not None
                else None
            ),
            "page_name": self.page_name,
            "source_object_ids": list(self.source_object_ids),
            "created_native_names": list(self.created_native_names),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DrawingEditResult:
    """Native drawing objects changed by one constrained edit."""

    created_native_names: tuple[str, ...] = ()
    changed_native_names: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "created_native_names": list(self.created_native_names),
            "changed_native_names": list(self.changed_native_names),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


class DrawingBackend(ABC):
    """CAD-specific implementation of native drawing creation and export."""

    @property
    @abstractmethod
    def supported_output_formats(self) -> frozenset[DrawingOutputFormat]:
        """Formats this CAD adapter can export without lossy conversion."""

    def validate_output_formats(
        self,
        output_formats: Optional[Sequence[str]],
    ) -> tuple[DrawingOutputFormat, ...]:
        """Normalize formats and fail explicitly for adapter limitations."""

        requested = normalize_output_formats(output_formats)
        unsupported = [
            item for item in requested if item not in self.supported_output_formats
        ]
        if unsupported:
            supported = ", ".join(sorted(self.supported_output_formats)) or "none"
            raise NotImplementedError(
                f"{type(self).__name__} does not support drawing export format(s): "
                f"{', '.join(unsupported)}. Supported formats: {supported}."
            )
        return requested

    @abstractmethod
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
        """Create a native drawing linked to ``objects`` and export requested files."""

    def inspect_drawing(
        self,
        *,
        document: CadDocument,
        page_name: str,
    ) -> dict[str, Any]:
        """List native views and editable drawing items."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support drawing inspection."
        )

    def list_drawings(self, *, document: CadDocument) -> list[dict[str, Any]]:
        """Discover existing native drawing pages in an opened document."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support drawing discovery."
        )

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
        """Add a geometry-measured native dimension for a semantic feature."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support feature dimensions."
        )

    def add_drawing_note(
        self,
        *,
        document: CadDocument,
        page_name: str,
        text: str,
        position_mm: tuple[float, float],
    ) -> DrawingEditResult:
        """Add an editorial note to an existing native drawing."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support drawing notes."
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
        """Add a native leader and associated annotation to a drawing view."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support drawing leaders."
        )

    def move_drawing_item(
        self,
        *,
        document: CadDocument,
        page_name: str,
        item_native_name: str,
        position_mm: tuple[float, float],
    ) -> DrawingEditResult:
        """Move an editable native drawing item on its page."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support moving drawing items."
        )

    def export_drawing(
        self,
        *,
        document: CadDocument,
        page_name: str,
        pdf_path: Path,
        vector_source_path: Path,
    ) -> DrawingEditResult:
        """Re-export an edited native drawing to its managed artifact paths."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support drawing re-export."
        )


class DrawingBackendFactory(ABC):
    """Factory for one CAD application's drawing backend."""

    @abstractmethod
    def create(self, document: CadDocument) -> DrawingBackend:
        """Create a drawing backend for ``document``."""


_FACTORIES: Dict[str, Type[DrawingBackendFactory]] = {}


def register_drawing_backend(
    backend_name: str,
    factory_type: Type[DrawingBackendFactory],
) -> None:
    """Register or replace the drawing factory for a backend."""

    normalized = backend_name.strip().lower()
    if not normalized:
        raise ValueError("backend_name must not be empty.")
    _FACTORIES[normalized] = factory_type


def create_drawing_backend(document: CadDocument) -> DrawingBackend:
    """Resolve the CAD-specific drawing backend for a native document."""

    backend_name = document.backend.strip().lower()
    if backend_name == "freecad" and backend_name not in _FACTORIES:
        from .integrations.freecad.drawing import FreeCADDrawingBackendFactory

        register_drawing_backend(backend_name, FreeCADDrawingBackendFactory)

    factory_type = _FACTORIES.get(backend_name)
    if factory_type is None:
        raise NotImplementedError(
            f"Technical drawing export is not available for {document.backend!r}."
        )
    return factory_type().create(document)


def finalize_vector_pdf(
    *,
    svg_path: str | Path,
    pdf_path: str | Path,
    page_width_mm: float = 420.0,
    page_height_mm: float = 297.0,
) -> dict[str, object]:
    """Convert a TechDraw SVG to a print PDF and verify embedded fonts.

    This deliberately runs outside the CAD GUI process. FreeCAD's bundled
    Python runtime does not include the Cairo PDF stack, while the MCP runtime
    does. The SVG remains vector content throughout the conversion.
    """

    source = Path(svg_path).expanduser().resolve(strict=True)
    destination = Path(pdf_path).expanduser().resolve()
    try:
        from weasyprint import CSS, HTML
    except ImportError as exc:
        raise RuntimeError(
            "Print-ready drawing conversion requires WeasyPrint."
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    html = (
        "<!doctype html><html><body>"
        f'<img src="{source.as_uri()}" alt="Technical drawing">'
        "</body></html>"
    )
    stylesheet = CSS(
        string=(
            f"@page {{ size: {page_width_mm}mm {page_height_mm}mm; margin: 0; }}"
            "html, body { margin: 0; padding: 0; width: 100%; height: 100%; "
            "overflow: hidden; background: white; }"
            "img { display: block; width: 100%; height: 100%; object-fit: fill; }"
        )
    )
    HTML(string=html, base_url=str(source.parent)).write_pdf(
        destination,
        stylesheets=[stylesheet],
        uncompressed_pdf=True,
    )
    return validate_print_ready_pdf(
        destination,
        page_width_mm=page_width_mm,
        page_height_mm=page_height_mm,
    )


def validate_print_ready_pdf(
    pdf_path: str | Path,
    *,
    page_width_mm: float = 420.0,
    page_height_mm: float = 297.0,
) -> dict[str, object]:
    """Validate physical page size and font embedding in a drawing PDF."""

    path = Path(pdf_path).expanduser().resolve(strict=True)
    try:
        from pypdf import PdfReader
    except ImportError:
        return _validate_uncompressed_pdf(
            path,
            page_width_mm=page_width_mm,
            page_height_mm=page_height_mm,
        )

    reader = PdfReader(path)
    if len(reader.pages) != 1:
        raise RuntimeError(
            f"Drawing PDF must have exactly one page, found {len(reader.pages)}."
        )
    page = reader.pages[0]
    width_mm = float(page.mediabox.width) * 25.4 / 72.0
    height_mm = float(page.mediabox.height) * 25.4 / 72.0
    if not (
        math.isclose(width_mm, page_width_mm, abs_tol=1.0)
        and math.isclose(height_mm, page_height_mm, abs_tol=1.0)
    ):
        raise RuntimeError(
            "Drawing PDF has an incorrect page size "
            f"({width_mm:.1f} x {height_mm:.1f} mm)."
        )

    fonts = _pdf_fonts(page.get("/Resources"))
    if not fonts:
        raise RuntimeError("Drawing PDF contains no inspectable font resources.")
    unembedded = [
        name for name, font in fonts.items() if not _pdf_font_is_embedded(font)
    ]
    if unembedded:
        raise RuntimeError(
            "Drawing PDF contains unembedded fonts: " + ", ".join(sorted(unembedded))
        )
    return {
        "page_width_mm": round(width_mm, 3),
        "page_height_mm": round(height_mm, 3),
        "font_count": len(fonts),
        "fonts_embedded": True,
        "vector_source": "techdraw_svg",
    }


def _validate_uncompressed_pdf(
    path: Path,
    *,
    page_width_mm: float,
    page_height_mm: float,
) -> dict[str, object]:
    """Validate the uncompressed PDF emitted by the bundled print pipeline."""

    data = path.read_bytes()
    if not data.startswith(b"%PDF-"):
        raise RuntimeError("Drawing output is not a PDF file.")
    page_counts = re.findall(
        rb"/Type\s*/Pages\b.*?/Count\s+(\d+)",
        data,
        flags=re.DOTALL,
    )
    if not page_counts or int(page_counts[0]) != 1:
        raise RuntimeError("Drawing PDF must have exactly one page.")
    media_box = re.search(
        rb"/MediaBox\s*\[\s*[-+0-9.]+\s+[-+0-9.]+\s+"
        rb"([-+0-9.]+)\s+([-+0-9.]+)\s*\]",
        data,
    )
    if media_box is None:
        raise RuntimeError("Drawing PDF has no inspectable media box.")
    width_mm = float(media_box.group(1)) * 25.4 / 72.0
    height_mm = float(media_box.group(2)) * 25.4 / 72.0
    if not (
        math.isclose(width_mm, page_width_mm, abs_tol=1.0)
        and math.isclose(height_mm, page_height_mm, abs_tol=1.0)
    ):
        raise RuntimeError(
            "Drawing PDF has an incorrect page size "
            f"({width_mm:.1f} x {height_mm:.1f} mm)."
        )

    descriptors = re.findall(
        rb"<<(?:(?!>>).)*?/Type\s*/FontDescriptor\b(?:(?!>>).)*?>>",
        data,
        flags=re.DOTALL,
    )
    if not descriptors:
        raise RuntimeError("Drawing PDF contains no inspectable font resources.")
    unembedded_count = sum(
        not re.search(rb"/FontFile(?:2|3)?\s+\d+\s+\d+\s+R\b", descriptor)
        for descriptor in descriptors
    )
    if unembedded_count:
        raise RuntimeError(
            f"Drawing PDF contains {unembedded_count} unembedded font(s)."
        )
    return {
        "page_width_mm": round(width_mm, 3),
        "page_height_mm": round(height_mm, 3),
        "font_count": len(descriptors),
        "fonts_embedded": True,
        "vector_source": "techdraw_svg",
        "validator": "builtin",
    }


def _pdf_fonts(resources: object) -> dict[str, object]:
    if resources is None:
        return {}
    get_object = getattr(resources, "get_object", None)
    resolved = get_object() if callable(get_object) else resources
    if not hasattr(resolved, "get"):
        return {}
    fonts: dict[str, object] = {}
    font_resources = resolved.get("/Font") or {}
    font_get_object = getattr(font_resources, "get_object", None)
    if callable(font_get_object):
        font_resources = font_get_object()
    for name, font in font_resources.items():
        font_object = getattr(font, "get_object", lambda: font)()
        fonts[str(name)] = font_object
    xobjects = resolved.get("/XObject") or {}
    xobject_get_object = getattr(xobjects, "get_object", None)
    if callable(xobject_get_object):
        xobjects = xobject_get_object()
    for name, xobject in xobjects.items():
        xobject_object = getattr(xobject, "get_object", lambda: xobject)()
        nested = _pdf_fonts(
            xobject_object.get("/Resources") if hasattr(xobject_object, "get") else None
        )
        fonts.update({f"{name}:{key}": value for key, value in nested.items()})
    return fonts


def _pdf_font_is_embedded(font: object) -> bool:
    if not hasattr(font, "get"):
        return False
    if str(font.get("/Subtype")) == "/Type3":
        return True
    descriptor = font.get("/FontDescriptor")
    if descriptor is None:
        descendants = font.get("/DescendantFonts") or ()
        for descendant in descendants:
            descendant_object = getattr(descendant, "get_object", lambda: descendant)()
            if _pdf_font_is_embedded(descendant_object):
                return True
        return False
    descriptor_object = getattr(descriptor, "get_object", lambda: descriptor)()
    return any(
        key in descriptor_object for key in ("/FontFile", "/FontFile2", "/FontFile3")
    )


__all__ = [
    "DimensionIntent",
    "DimensionKind",
    "DimensionPlacement",
    "DimensionSide",
    "DimensionView",
    "DrawingBackend",
    "DrawingBackendFactory",
    "DrawingOutputFormat",
    "DrawingResult",
    "DrawingStandard",
    "DrawingViewSpec",
    "DRAWING_OUTPUT_FORMATS",
    "ProjectionAngle",
    "create_drawing_backend",
    "finalize_vector_pdf",
    "format_dimension_value",
    "normalize_drawing_standard",
    "normalize_projection_angle",
    "normalize_output_formats",
    "projected_view_layout",
    "register_drawing_backend",
    "select_drawing_scale",
    "validate_print_ready_pdf",
]
