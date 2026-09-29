"""Stateful RapidCADPy session for local and live CAD applications.

RPC-dispatch note (tracked follow-up, not fixed here): every "worker" call
below (``self._worker.call(method, params)``) and the FreeCAD GUI bridge
(``integrations/freecad/gui_bridge.py``) independently reflect on a live
``CadSession`` instance via ``getattr(session, method)(**params)`` — three
separately-loaded copies of this class (this process, the out-of-process
worker subprocess, and the FreeCAD GUI process), with no shared interface
or schema between them. Adding a method here only reaches the GUI bridge
once the FreeCAD connector is reinstalled *and* FreeCAD is restarted (see
``rapidcadpy/integrations/freecad/connector_addon.py``); until then callers
get a runtime "Unknown GUI bridge/worker method" error instead of a
load-time failure. A real fix means introducing an explicit RPC/backend
contract these three dispatchers can be checked against, rather than
reflection over whatever public methods happen to exist.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .application_discovery import ApplicationDiscovery
from .cad_objects import CadDocument, CadObject, CadParameter
from .document_service import DocumentService
from .drawing_service import DrawingService
from .feature import Feature
from .feature_updates import LoftFeatureUpdate, SweepFeatureUpdate
from .integrations.freecad.capabilities import FREECAD_CAPABILITIES
from .integrations.freecad.document_hydrator import FreeCADDocumentCodec
from .live_backend import LiveCadBackend, LiveCadBackendRegistry
from .modeling import (
    CadPath,
    CadProfile,
    ModelingRequest,
    PathDefinition,
    ProfileDefinition,
)
from .modeling_service import ModelingService
from .mutations import MutationCoordinator
from .operation_result import CadOperationResult, CadOperationSupport
from .parameter_service import ParameterService
from .session_records import OperationRecord, SemanticObject


class CadSession:
    """Stateful standalone API around one live RapidCADPy CAD document."""

    _FREECAD_CAPABILITIES = FREECAD_CAPABILITIES

    LIVE_CONTRACT_VERSION = (1, 0)

    def __init__(
        self,
        execution_mode: str = "headless",
        app_factory: Optional[Callable[[str], Any]] = None,
        *,
        live_backend_registry: Optional[LiveCadBackendRegistry] = None,
    ) -> None:
        if execution_mode not in {"headless", "gui", "embedded"}:
            raise ValueError("execution_mode must be 'headless', 'gui', or 'embedded'.")
        self.execution_mode = execution_mode
        self._live_backend_registry = live_backend_registry or LiveCadBackendRegistry()
        self._live_backend: Optional[LiveCadBackend] = None
        self.profile_references: dict[str, CadProfile] = {}
        self.path_references: dict[str, CadPath] = {}
        self._profile_cache: dict[str, str] = {}
        # Constructs a backend App given a document name. Defaults to
        # FreeCADApp (imported lazily at each call site, matching prior
        # behavior); inject a different factory to run CadSession against
        # another App implementation instead of FreeCAD.
        self._app_factory: Optional[Callable[[str], Any]] = app_factory
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
        self._mutation_coordinator = MutationCoordinator(self)
        self._application_discovery = ApplicationDiscovery(self)
        self._document_service = DocumentService(self)
        self._modeling_service = ModelingService(self)
        self._parameter_service = ParameterService(self)
        self._drawing_service = DrawingService(self)
        self._freecad_codec = FreeCADDocumentCodec(self)

    @property
    def application_discovery(self) -> ApplicationDiscovery:
        """Application discovery and connection lifecycle service."""
        return self._application_discovery

    @property
    def document_service(self) -> DocumentService:
        """Document lifecycle and session inspection service."""
        return self._document_service

    @property
    def modeling_service(self) -> ModelingService:
        """Sketching, feature creation, and export service."""
        return self._modeling_service

    @property
    def parameter_service(self) -> ParameterService:
        """Named parameter and binding service."""
        return self._parameter_service

    @property
    def drawing_service(self) -> DrawingService:
        """Technical drawing lifecycle service."""
        return self._drawing_service

    @property
    def mutation_coordinator(self) -> MutationCoordinator:
        """Revision-safe native mutation coordinator."""
        return self._mutation_coordinator

    def setup_backend(
        self, cad_system: str = "freecad", document_name: str = "RapidCADPy"
    ) -> Dict[str, Any]:
        return self._application_discovery.setup_backend(cad_system, document_name)

    def launch_freecad_gui(self) -> Dict[str, Any]:
        """Launch one bridge-enabled FreeCAD GUI and retain its connection."""
        return self._application_discovery.launch_freecad_gui()

    def list_cad_applications(self, software: Optional[str] = None) -> Dict[str, Any]:
        """Discover applications through registered live integration factories."""
        return self._application_discovery.list_cad_applications(software)

    def select_cad_application(
        self, software: str, target_id: Optional[str] = None, use_active_document: bool = True
    ) -> Dict[str, Any]:
        """Select a compatible family of live services from one integration."""
        return self._application_discovery.select_cad_application(
            software, target_id, use_active_document
        )

    @property
    def live_backend(self) -> Optional[LiveCadBackend]:
        """Selected service family for runtime composition, never tool serialization."""
        return self._application_discovery.live_backend

    def inspect_modeling_support(
        self, definition: ModelingRequest | dict[str, Any]
    ) -> Dict[str, Any]:
        """Inspect the new modeling contract without mutating the document."""
        return self._application_discovery.inspect_modeling_support(definition)

    def apply_modeling(
        self, definition: ModelingRequest | dict[str, Any], *, expected_revision: str
    ) -> Dict[str, Any]:
        """Apply shared intent through the selected live modeling service."""
        return self._application_discovery.apply_modeling(
            definition, expected_revision=expected_revision
        )

    def get_active_cad_application(self) -> Dict[str, Any]:
        """Describe the currently selected backend-neutral CAD target."""
        return self._application_discovery.get_active_cad_application()

    def install_cad_connector(
        self, software: str, install_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Install the connector for one supported CAD application."""
        return self._application_discovery.install_cad_connector(software, install_dir)

    def list_freecad_instances(self) -> Dict[str, Any]:
        """List running FreeCAD GUIs that advertise a RapidCADPy bridge."""
        return self._application_discovery.list_freecad_instances()

    def install_freecad_connector(self, mod_dir: Optional[str] = None) -> Dict[str, Any]:
        """Install RapidCADPy's auto-start bridge into FreeCAD's user modules."""
        return self._application_discovery.install_freecad_connector(mod_dir)

    def attach_freecad(
        self, instance_id: Optional[str] = None, use_active_document: bool = True
    ) -> Dict[str, Any]:
        """Attach to an addon-enabled FreeCAD GUI and optionally hydrate its document."""
        return self._application_discovery.attach_freecad(instance_id, use_active_document)

    def use_active_document(self) -> Dict[str, Any]:
        """Hydrate the document currently active in an attached FreeCAD GUI."""
        return self._document_service.use_active_document()

    def new_document(self, name: str = "RapidCADPy") -> Dict[str, Any]:
        return self._document_service.new_document(name)

    def open_document(self, path: str) -> Dict[str, Any]:
        """Open and hydrate a native FreeCAD document into the live session."""
        return self._document_service.open_document(path)

    def execute_code(self, code: str, allow_direct_geometry: bool = False) -> Dict[str, Any]:
        """Execute code in the GUI, rejecting new baked features by default."""
        return self._document_service.execute_code(code, allow_direct_geometry)

    def work_plane(
        self,
        plane: str = "XY",
        offset: Optional[float] = None,
        origin: Optional[List[float]] = None,
        normal: Optional[List[float]] = None,
        x_axis: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        return self._modeling_service.work_plane(plane, offset, origin, normal, x_axis)

    def move_to(self, x: float, y: float) -> Dict[str, Any]:
        return self._modeling_service.move_to(x, y)

    def line_to(self, x: float, y: float) -> Dict[str, Any]:
        return self._modeling_service.line_to(x, y)

    def three_point_arc(
        self, mid_x: float, mid_y: float, end_x: float, end_y: float
    ) -> Dict[str, Any]:
        return self._modeling_service.three_point_arc(mid_x, mid_y, end_x, end_y)

    def rect(self, width: float, height: float, centered: bool = True) -> Dict[str, Any]:
        return self._modeling_service.rect(width, height, centered)

    def circle(self, radius: float) -> Dict[str, Any]:
        return self._modeling_service.circle(radius)

    def list_profiles(self) -> Dict[str, Any]:
        return self._modeling_service.list_profiles()

    def list_item_angle_brackets(self) -> Dict[str, Any]:
        """List standardized ITEM angle-bracket catalog entries."""
        return self._modeling_service.list_item_angle_brackets()

    def get_item_angle_bracket(self, name: str) -> Dict[str, Any]:
        """Return one standardized ITEM angle-bracket catalog entry."""
        return self._modeling_service.get_item_angle_bracket(name)

    def sketch_profile(
        self, family: str, name: str, x: float = 0.0, y: float = 0.0
    ) -> Dict[str, Any]:
        return self._modeling_service.sketch_profile(family, name, x, y)

    def box(
        self, length: float, width: float, height: float, centered: bool = True
    ) -> Dict[str, Any]:
        return self._modeling_service.box(length, width, height, centered)

    def extrude(
        self,
        distance: float,
        operation: str = "new_body",
        symmetric: bool = False,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._modeling_service.extrude(distance, operation, symmetric, expected_revision)

    def _active_extrusion_path(self, *, distance: float, symmetric: bool) -> Dict[str, Any]:
        return self._modeling_service._active_extrusion_path(distance=distance, symmetric=symmetric)

    def _profile_operation(self, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        return self._modeling_service._profile_operation(operation, params)

    def create_profile(
        self,
        workplane_id: str | None = None,
        expected_revision: str | None = None,
        *,
        definition: ProfileDefinition | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create typed geometry or capture existing workplane geometry."""
        return self._modeling_service.create_profile(
            workplane_id, expected_revision, definition=definition
        )

    def create_path(
        self,
        workplane_id: str | None = None,
        expected_revision: str | None = None,
        *,
        definition: PathDefinition | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a typed planar spine or capture an existing workplane wire."""
        return self._modeling_service.create_path(
            workplane_id, expected_revision, definition=definition
        )

    def _create_modeling_geometry(
        self,
        definition: ProfileDefinition | PathDefinition | dict[str, Any],
        kind: str,
        expected_revision: str | None,
    ) -> dict[str, Any]:
        return self._modeling_service._create_modeling_geometry(definition, kind, expected_revision)

    def update_profile(
        self,
        profile_id: str,
        workplane_id: str | None = None,
        expected_revision: str | None = None,
        *,
        definition: ProfileDefinition | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update the same native sketch from typed geometry or a workplane."""
        return self._modeling_service.update_profile(
            profile_id, workplane_id, expected_revision, definition=definition
        )

    def update_feature(
        self,
        feature_id: str,
        parameters: LoftFeatureUpdate | SweepFeatureUpdate | dict[str, Any],
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Edit declared native loft/sweep parameters without replacing a feature."""
        return self._modeling_service.update_feature(feature_id, parameters, expected_revision)

    def inspect_capabilities(self) -> dict[str, Any]:
        """Inspect the selected integration's supported operations and restrictions."""
        return self._modeling_service.inspect_capabilities()

    def inspect_geometry(self, object_ids: list[str] | None = None) -> dict[str, Any]:
        """Report native validity, dimensions and dependencies without handles."""
        return self._modeling_service.inspect_geometry(object_ids)

    def loft(
        self,
        profile_workplane_ids: list[str] | None = None,
        make_solid: bool = True,
        ruled: bool = False,
        expected_revision: str | None = None,
        *,
        profile_ids: list[str] | None = None,
        alignment: str = "automatic",
    ) -> dict[str, Any]:
        """Loft linked profiles; legacy workplane arguments capture reusable sketches."""
        return self._modeling_service.loft(
            profile_workplane_ids,
            make_solid,
            ruled,
            expected_revision,
            profile_ids=profile_ids,
            alignment=alignment,
        )

    def sweep(
        self,
        profile_id: str,
        path_id: str,
        make_solid: bool = True,
        orientation: str = "corrected_frenet",
        transition: str = "right",
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Sweep a linked closed section along an open planar sketch spine."""
        return self._modeling_service.sweep(
            profile_id, path_id, make_solid, orientation, transition, expected_revision
        )

    def update_path(
        self,
        path_id: str,
        definition: PathDefinition | dict[str, Any],
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Replace an unconstrained planar path in place and recompute dependents."""
        return self._modeling_service.update_path(path_id, definition, expected_revision)

    def cut(self, target_id: str, tool_id: str) -> Dict[str, Any]:
        return self._modeling_service.cut(target_id, tool_id)

    def union(
        self, target_id: str, tool_ids: List[str], expected_revision: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._modeling_service.union(target_id, tool_ids, expected_revision)

    def fillet(
        self, shape_id: Optional[str] = None, radius: float = 1.0, selector: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._modeling_service.fillet(shape_id, radius, selector)

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
        return self._modeling_service.hole(
            target_id,
            diameter,
            center,
            axis,
            through,
            depth,
            hole_type,
            countersink_diameter,
            countersink_angle,
            expected_revision,
        )

    def apply_feature(
        self, feature: "Feature", target_id: str, expected_revision: Optional[str] = None
    ) -> Dict[str, Any]:
        """Apply a semantic feature to a hydrated native CAD object.

        This is the generic mutation path for objects that were created outside
        RapidCADPy as well as for objects created through the fluent API.
        """
        return self._modeling_service.apply_feature(feature, target_id, expected_revision)

    @staticmethod
    def _vector3(value: list[float], name: str) -> tuple[float, float, float]:
        if len(value) != 3:
            raise ValueError(f"{name} must contain exactly three coordinates.")
        return (float(value[0]), float(value[1]), float(value[2]))

    def export_step(self, path: str, shape_id: Optional[str] = None) -> Dict[str, Any]:
        return self._modeling_service.export_step(path, shape_id)

    def export_stl(self, path: str, shape_id: Optional[str] = None) -> Dict[str, Any]:
        return self._modeling_service.export_stl(path, shape_id)

    def export_native(self, path: str, shape_id: Optional[str] = None) -> Dict[str, Any]:
        return self._modeling_service.export_native(path, shape_id)

    def describe_freecad_file(self, path: str) -> Dict[str, Any]:
        return self._document_service.describe_freecad_file(path)

    def render(
        self,
        path: str,
        view: str = "iso",
        shape_id: Optional[str] = None,
        width: int = 1000,
        height: int = 800,
    ) -> Dict[str, Any]:
        return self._document_service.render(path, view, shape_id, width, height)

    def _render_response(
        self,
        path: Path,
        view: str,
        shape_id: Optional[str],
        width: int,
        height: int,
        operation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._document_service._render_response(
            path, view, shape_id, width, height, operation_id
        )

    def add_parameter(self, name: str, value: float, units: str = "mm") -> Dict[str, Any]:
        """Compatibility wrapper for the persistent named-parameter API."""
        return self._parameter_service.add_parameter(name, value, units)

    def list_parameters(self) -> Dict[str, Any]:
        """List persistent named parameters in the active native document."""
        return self._parameter_service.list_parameters()

    def get_parameter(self, parameter_id: str) -> Dict[str, Any]:
        """Inspect one persistent named parameter by RapidCAD ID or name."""
        return self._parameter_service.get_parameter(parameter_id)

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
        return self._parameter_service.create_parameter(
            name, parameter_type, value, unit, expression, expected_revision
        )

    def set_parameters(
        self, updates: list[Dict[str, Any]], expected_revision: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update named parameters atomically and recompute the native document."""
        return self._parameter_service.set_parameters(updates, expected_revision)

    def bind_parameter(
        self,
        parameter_id: str,
        object_id: str,
        property_name: str,
        expression: Optional[str] = None,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Bind a named parameter expression to a generic native feature property."""
        return self._parameter_service.bind_parameter(
            parameter_id, object_id, property_name, expression, expected_revision
        )

    def describe_state(self) -> Dict[str, Any]:
        return self._document_service.describe_state()

    def list_objects(self) -> Dict[str, Any]:
        return self._document_service.list_objects()

    def get_object(self, object_id: str) -> Dict[str, Any]:
        """Return one semantic object by its RapidCAD ID."""
        return self._document_service.get_object(object_id)

    def _semantic_features_for_object(self, object_id: str) -> list[Dict[str, Any]]:
        return self._document_service._semantic_features_for_object(object_id)

    @staticmethod
    def _drawing_dimension_requests(feature: Dict[str, Any]) -> list[Dict[str, Any]]:
        return DrawingService._drawing_dimension_requests(feature)

    def set_object_property(
        self,
        object_id: str,
        property_name: str,
        value: Any,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Modify a property on the same live native CAD object."""
        return self._document_service.set_object_property(
            object_id, property_name, value, expected_revision
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
        return self._drawing_service.generate_drawing(
            object_ids,
            standard,
            sheet_size,
            projection_angle,
            template_id,
            output_directory,
            part_name,
            run_id,
            include_native,
            dimension_feature_ids,
            output_formats,
            expected_revision,
        )

    def get_current_drawing(self) -> Dict[str, Any]:
        """Return the live session's latest drawing workspace manifest."""
        return self._drawing_service.get_current_drawing()

    def list_drawings(self) -> Dict[str, Any]:
        """Discover native drawing pages in the currently opened CAD file."""
        return self._drawing_service.list_drawings()

    def select_drawing(self, drawing_id: str) -> Dict[str, Any]:
        """Select one discovered drawing page for subsequent editing tools."""
        return self._drawing_service.select_drawing(drawing_id)

    def list_drawing_items(self, drawing_id: Optional[str] = None) -> Dict[str, Any]:
        """List exact view and editable-item IDs on an existing drawing."""
        return self._drawing_service.list_drawing_items(drawing_id)

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
        return self._drawing_service.add_feature_dimension(
            feature_id, dimension_kind, view_id, position_mm, drawing_id, expected_revision
        )

    def add_drawing_note(
        self,
        text: str,
        position_mm: list[float],
        drawing_id: Optional[str] = None,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add an editorial native TechDraw annotation."""
        return self._drawing_service.add_drawing_note(
            text, position_mm, drawing_id, expected_revision
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
        return self._drawing_service.add_drawing_leader(
            text,
            view_name,
            view_id,
            anchor_mm,
            elbow_mm,
            text_position_mm,
            drawing_id,
            expected_revision,
        )

    def move_drawing_item(
        self,
        item_id: str,
        position_mm: list[float],
        drawing_id: Optional[str] = None,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Move an existing RapidCAD dimension, note, or leader."""
        return self._drawing_service.move_drawing_item(
            item_id, position_mm, drawing_id, expected_revision
        )

    def export_current_drawing(
        self, drawing_id: Optional[str] = None, expected_revision: Optional[str] = None
    ) -> Dict[str, Any]:
        """Re-export the edited page to its managed PDF and SVG paths."""
        return self._drawing_service.export_current_drawing(drawing_id, expected_revision)

    def _resolve_drawing(self, drawing_id: Optional[str]) -> Dict[str, Any]:
        return self._drawing_service._resolve_drawing(drawing_id)

    def _discover_drawings(self) -> list[Dict[str, Any]] | Dict[str, Any]:
        return self._drawing_service._discover_drawings()

    def _prepare_drawing_edit(
        self, drawing_id: Optional[str], expected_revision: Optional[str]
    ) -> Dict[str, Any]:
        return self._drawing_service._prepare_drawing_edit(drawing_id, expected_revision)

    def _drawing_runtime_object(
        self, object_id: str, expected_kind: str
    ) -> CadObject | Dict[str, Any]:
        return self._drawing_service._drawing_runtime_object(object_id, expected_kind)

    @staticmethod
    def _drawing_view_lookup(views: list[Dict[str, Any]]) -> Dict[str, str]:
        return DrawingService._drawing_view_lookup(views)

    def _resolve_drawing_view(
        self, drawing: Dict[str, Any], *, view_name: Optional[str], view_id: Optional[str]
    ) -> CadObject | Dict[str, Any]:
        return self._drawing_service._resolve_drawing_view(
            drawing, view_name=view_name, view_id=view_id
        )

    @staticmethod
    def _drawing_position(value: list[float]) -> tuple[float, float]:
        return DrawingService._drawing_position(value)

    def _complete_drawing_edit(
        self,
        drawing: Dict[str, Any],
        operation: str,
        params: Dict[str, Any],
        edit: Any,
        summary: str,
    ) -> Dict[str, Any]:
        return self._drawing_service._complete_drawing_edit(
            drawing, operation, params, edit, summary
        )

    def save_document(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Save the active native document, optionally to a new path."""
        return self._document_service.save_document(path)

    def select_object(self, object_id: str) -> Dict[str, Any]:
        """Select a native object in the connected CAD GUI."""
        return self._document_service.select_object(object_id)

    def fit_view(self) -> Dict[str, Any]:
        """Fit the connected CAD GUI view to the active model."""
        return self._document_service.fit_view()

    def export_screenshot(
        self,
        path: Optional[str] = None,
        view: str = "isometric",
        width: int = 1024,
        height: int = 768,
        fit: bool = True,
    ) -> Dict[str, Any]:
        """Capture the active native viewport and optionally write it as PNG."""
        return self._document_service.export_screenshot(path, view, width, height, fit)

    @staticmethod
    def _write_screenshot(path: str, content: bytes) -> Path:
        return DocumentService._write_screenshot(path, content)

    def get_history(self, limit: int = 50) -> Dict[str, Any]:
        return self._document_service.get_history(limit)

    def _export(self, path: str, kind: str, shape_id: Optional[str] = None) -> Dict[str, Any]:
        return self._modeling_service._export(path, kind, shape_id)

    def _geometry_signature(self, shape: Any) -> Dict[str, Any]:
        return self._freecad_codec._geometry_signature(shape)

    def _hydrate_freecad_document(
        self, doc: Any, file_path: Optional[Path], source_tool: str = "open_document"
    ) -> Dict[str, Any]:
        return self._freecad_codec._hydrate_freecad_document(doc, file_path, source_tool)

    def _freecad_properties(self, obj: Any, ids_by_name: Dict[str, str]) -> Dict[str, Any]:
        return self._freecad_codec._freecad_properties(obj, ids_by_name)

    def _serialize_freecad_value(
        self, value: Any, ids_by_name: Dict[str, str], depth: int = 0
    ) -> Any:
        return self._freecad_codec._serialize_freecad_value(value, ids_by_name, depth)

    def _freecad_object_ids(self, native_objects: Any, ids_by_name: Dict[str, str]) -> list[str]:
        return self._freecad_codec._freecad_object_ids(native_objects, ids_by_name)

    def _freecad_containment(
        self, native_objects: list[Any], ids_by_name: Dict[str, str]
    ) -> Dict[str, Dict[str, list[str]]]:
        return self._freecad_codec._freecad_containment(native_objects, ids_by_name)

    def _freecad_tree(
        self, object_ids: list[str], children: Dict[str, list[str]], parents: Dict[str, list[str]]
    ) -> list[Dict[str, Any]]:
        return self._freecad_codec._freecad_tree(object_ids, children, parents)

    def _freecad_semantic_type(self, obj: Any) -> str:
        return self._freecad_codec._freecad_semantic_type(obj)

    def _find_parameter(self, identifier: str) -> Optional[CadParameter]:
        return self._parameter_service._find_parameter(identifier)

    def _parameter_context(self) -> tuple[Any, Any]:
        return self._parameter_service._parameter_context()

    def _parameter_revision_error(
        self, expected_revision: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        return self._parameter_service._parameter_revision_error(expected_revision)

    def _rehydrate_parameter_document(self, source_tool: str) -> Dict[str, Any]:
        return self._parameter_service._rehydrate_parameter_document(source_tool)

    def _recalculate_document_revision(self) -> str:
        return self._parameter_service._recalculate_document_revision()

    def _describe_freecad_object(self, obj: Any) -> Dict[str, Any]:
        return self._freecad_codec._describe_freecad_object(obj)

    def _shape_signature(self, obj: Any) -> Dict[str, Any]:
        return self._freecad_codec._shape_signature(obj)

    def _record(self, tool: str, args: Dict[str, Any], outputs: list[str], summary: str) -> str:
        op_id = self._new_id("op")
        self.operations.append(
            OperationRecord(id=op_id, tool=tool, args=args, outputs=outputs, summary=summary)
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
        if self.execution_mode != "gui" or self._worker is not None or self.app is not None:
            return None

        attached = self.attach_freecad(use_active_document=require_document)
        if not attached.get("ok"):
            return attached
        if self._worker is None:
            return self._error("FreeCAD attachment did not create a live GUI connection.")
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
            f"Attached to FreeCAD but could not hydrate its active document: {attached['warning']}"
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
            return self._error("No backend configured. Call setup_backend('freecad') first.")
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

    def _operation_ok(
        self,
        summary: str,
        *,
        document_revision: Optional[str] = None,
        created_object_ids: Optional[List[str]] = None,
        changed_object_ids: Optional[List[str]] = None,
        removed_object_ids: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        support: Optional[CadOperationSupport] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        return CadOperationResult.success(
            summary,
            document_revision=document_revision,
            created_object_ids=tuple(created_object_ids or ()),
            changed_object_ids=tuple(changed_object_ids or ()),
            removed_object_ids=tuple(removed_object_ids or ()),
            warnings=tuple(warnings or ()),
            support=support,
            data=extra,
        ).to_dict()

    def _operation_error(
        self,
        message: str,
        *,
        error_code: str = "cad_operation_failed",
        **extra: Any,
    ) -> Dict[str, Any]:
        return CadOperationResult.failure(
            message,
            error_code=error_code,
            document_revision=self.document_revision,
            data=extra,
        ).to_dict()

    def _error(self, message: str, *, error_code: str = "cad_operation_failed") -> Dict[str, Any]:
        return CadOperationResult.failure(message, error_code=error_code).to_dict()

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
