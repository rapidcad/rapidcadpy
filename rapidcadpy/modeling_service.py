"""Focused ModelingService operations composed by :class:`CadSession`."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cad_objects import CadObject
from .feature import Feature
from .feature_updates import LoftFeatureUpdate, SweepFeatureUpdate
from .modeling import (
    PathDefinition,
    ProfileDefinition,
    modeling_request_from_dict,
)
from .operation_result import CadOperationSupport
from .session_records import SemanticObject
from .session_service import SessionService


class ModelingService(SessionService):
    def work_plane(
        self,
        plane: str = "XY",
        offset: Optional[float] = None,
        origin: Optional[List[float]] = None,
        normal: Optional[List[float]] = None,
        x_axis: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "work_plane",
                {
                    "plane": plane,
                    "offset": offset,
                    "origin": origin,
                    "normal": normal,
                    "x_axis": x_axis,
                },
            )
        ready = self._require_app()
        if ready is not None:
            return ready
        placed = origin is not None and normal is not None
        try:
            wp = self.app.work_plane(
                plane,
                offset=offset,
                origin=origin,
                normal=normal,
                x_axis=x_axis,
            )
        except Exception as exc:
            return self._error(f"Could not create workplane: {type(exc).__name__}: {exc}")

        wp_id = self._new_id("workplane")
        if placed:
            label = f"workplane at origin {origin}, normal {normal}"
            if x_axis is not None:
                label += f", x-axis {x_axis}"
        else:
            label = f"{plane.upper()} workplane"
        op_id = self._record(
            "work_plane",
            {
                "plane": plane,
                "offset": offset,
                "origin": origin,
                "normal": normal,
                "x_axis": x_axis,
            },
            [wp_id],
            f"Created {label}",
        )
        self.active_workplane = wp
        self.active_workplane_id = wp_id
        self.runtime_objects[wp_id] = wp
        self.objects[wp_id] = SemanticObject(
            id=wp_id,
            type="workplane",
            label=label,
            source_op=op_id,
            metadata={
                "plane": plane.upper(),
                "offset": offset,
                "origin": origin,
                "normal": normal,
                "x_axis": x_axis,
            },
        )
        return self._ok(
            summary=f"Created {label}",
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
        op_id = self._record("move_to", {"x": x, "y": y}, [], f"Moved sketch cursor to ({x}, {y})")
        return self._ok(summary=f"Moved sketch cursor to ({x}, {y})", operation_id=op_id)

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
        op_id = self._record("line_to", {"x": x, "y": y}, [], f"Drew line to ({x}, {y})")
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

    def rect(self, width: float, height: float, centered: bool = True) -> Dict[str, Any]:
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
            self.active_workplane.rect(float(width), float(height), centered=bool(centered))
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

    def list_profiles(self) -> Dict[str, Any]:
        from .components import profiles

        families = profiles.list_profiles()
        return self._ok(
            summary="Listed available preset profile families",
            profiles=families,
        )

    def list_item_angle_brackets(self) -> Dict[str, Any]:
        """List standardized ITEM angle-bracket catalog entries."""
        from .components.item import item_angle_bracket, list_item_angle_brackets

        brackets = [asdict(item_angle_bracket(name)) for name in list_item_angle_brackets()]
        return self._ok(
            summary="Listed available ITEM angle brackets",
            brackets=brackets,
        )

    def get_item_angle_bracket(self, name: str) -> Dict[str, Any]:
        """Return one standardized ITEM angle-bracket catalog entry."""
        from .components.item import item_angle_bracket

        try:
            bracket = item_angle_bracket(name)
        except (AttributeError, TypeError, ValueError) as exc:
            return self._error(str(exc))
        return self._ok(
            summary=f"Loaded ITEM angle bracket {bracket.name}",
            bracket=asdict(bracket),
        )

    def sketch_profile(
        self, family: str, name: str, x: float = 0.0, y: float = 0.0
    ) -> Dict[str, Any]:
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "sketch_profile", {"family": family, "name": name, "x": x, "y": y}
            )
        ready = self._require_workplane()
        if ready is not None:
            return ready

        from .components import profiles

        family_key = family.strip().lower()
        factories = {"ipe": profiles.ipe, "ipn": profiles.ipn, "item": profiles.item}
        factory = factories.get(family_key)
        if factory is None:
            return self._error(
                f"Unknown profile family {family!r}. Available: {', '.join(sorted(factories))}"
            )
        try:
            section = factory(name)
            section.sketch(self.active_workplane, x=float(x), y=float(y))
        except Exception as exc:
            return self._error(f"sketch_profile failed: {type(exc).__name__}: {exc}")

        profile_id = self._new_id("profile")
        op_id = self._record(
            "sketch_profile",
            {"family": family_key, "name": section.name, "x": x, "y": y},
            [profile_id],
            f"Drew {family_key.upper()} profile {section.name}",
        )
        self.objects[profile_id] = SemanticObject(
            id=profile_id,
            type="profile",
            label=f"{family_key}:{section.name}",
            source_op=op_id,
            metadata={
                "family": family_key,
                "name": section.name,
                "x": float(x),
                "y": float(y),
            },
        )
        return self._ok(
            summary=f"Drew {family_key.upper()} profile {section.name}",
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
        self,
        distance: float,
        operation: str = "new_body",
        symmetric: bool = False,
        expected_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(
                "extrude",
                {
                    "distance": distance,
                    "operation": operation,
                    "symmetric": symmetric,
                    "expected_revision": expected_revision,
                },
            )
        ready = self._require_workplane()
        if ready is not None:
            return ready
        if expected_revision and expected_revision != self.document_revision:
            return self._operation_error(
                "Document revision mismatch: expected "
                f"{expected_revision}, current {self.document_revision}.",
                error_code="document_revision_mismatch",
            )
        op_map = {
            "new_body": "NewBodyFeatureOperation",
            "join": "JoinBodyFeatureOperation",
            "cut": "Cut",
            "intersect": "Intersect",
        }
        backend_operation = op_map.get(operation, operation)
        try:
            extrusion_path = self._active_extrusion_path(
                distance=float(distance),
                symmetric=bool(symmetric),
            )
        except Exception as exc:
            return self._error(f"Could not resolve extrusion path: {type(exc).__name__}: {exc}")
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
            {
                "distance": distance,
                "operation": operation,
                "symmetric": symmetric,
                "extrusion_path": extrusion_path,
            },
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
                "extrusion_path": extrusion_path,
            },
        )
        self._recalculate_document_revision()
        return self._operation_ok(
            summary=f"Extruded active profile by {distance}",
            document_revision=self.document_revision,
            created_object_ids=[shape_id],
            support=CadOperationSupport(
                operation="extrude",
                status="supported",
                mode="native_feature",
                reason="Created a live native extrusion feature.",
            ),
            operation_id=op_id,
            object_id=shape_id,
            active_shape_id=shape_id,
            geometry=signature,
            extrusion_path=extrusion_path,
        )

    def _active_extrusion_path(
        self,
        *,
        distance: float,
        symmetric: bool,
    ) -> Dict[str, Any]:
        """Describe an extrusion's reference line in world coordinates."""

        workplane = self.active_workplane
        origin = tuple(float(value) for value in workplane._to_3d(0.0, 0.0))
        normal_value = getattr(workplane, "normal_vector", None)
        if normal_value is None:
            normal_value = getattr(workplane, "normal", None)
        if normal_value is None:
            raise ValueError("Active workplane does not expose its normal.")

        normal = tuple(float(value) for value in normal_value)
        normal_length = sum(value * value for value in normal) ** 0.5
        if normal_length <= 1e-12:
            raise ValueError("Active workplane normal must be non-zero.")
        unit_normal = tuple(value / normal_length for value in normal)

        if symmetric:
            start = tuple(origin[index] - unit_normal[index] * distance / 2.0 for index in range(3))
            end = tuple(origin[index] + unit_normal[index] * distance / 2.0 for index in range(3))
        else:
            start = origin
            end = tuple(origin[index] + unit_normal[index] * distance for index in range(3))

        vector = tuple(end[index] - start[index] for index in range(3))
        length = sum(value * value for value in vector) ** 0.5
        direction = tuple(value / length for value in vector) if length > 1e-12 else unit_normal

        def serialize(values: tuple[float, ...]) -> list[float]:
            return [0.0 if abs(value) <= 1e-12 else float(value) for value in values]

        return {
            "reference": "active_workplane_origin",
            "workplane_id": self.active_workplane_id,
            "start": serialize(start),
            "end": serialize(end),
            "vector": serialize(vector),
            "direction": serialize(direction),
            "distance": abs(float(distance)),
            "signed_distance": float(distance),
            "workplane_normal": serialize(unit_normal),
            "symmetric": bool(symmetric),
        }

    def _profile_operation(self, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call(operation, params)
        if self.cad_document is None:
            return self._error("No native CAD document is open.")
        factory = getattr(self.cad_document.adapter, "profile_operations", None)
        if not callable(factory):
            return self._operation_error(
                "Selected backend does not support persistent profile operations.",
                error_code="cad_operation_not_supported",
            )
        try:
            operations = factory(self)
            if operation == "loft":
                return operations.loft(
                    profile_ids=params.get("profile_ids"),
                    workplane_ids=params.get("profile_workplane_ids"),
                    make_solid=params["make_solid"],
                    ruled=params["ruled"],
                    expected_revision=params.get("expected_revision"),
                    **({"alignment": params["alignment"]} if "alignment" in params else {}),
                )
            if operation == "create_path":
                return operations.create_profile(**params, kind="path")
            return getattr(operations, operation)(**params)
        except Exception as exc:
            return self._error(f"{operation} failed: {type(exc).__name__}: {exc}")

    def create_profile(
        self,
        workplane_id: str | None = None,
        expected_revision: str | None = None,
        *,
        definition: ProfileDefinition | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create typed geometry or capture existing workplane geometry."""
        if definition is not None:
            if workplane_id is not None:
                return self._error("Specify a definition or workplane_id, exclusively.")
            return self._create_modeling_geometry(definition, "profile", expected_revision)
        return self._profile_operation(
            "create_profile",
            {
                "workplane_id": workplane_id,
                "expected_revision": expected_revision,
            },
        )

    def create_path(
        self,
        workplane_id: str | None = None,
        expected_revision: str | None = None,
        *,
        definition: PathDefinition | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a typed planar spine or capture an existing workplane wire."""
        if definition is not None:
            if workplane_id is not None:
                return self._error("Specify a definition or workplane_id, exclusively.")
            return self._create_modeling_geometry(definition, "path", expected_revision)
        return self._profile_operation(
            "create_path",
            {
                "workplane_id": workplane_id,
                "expected_revision": expected_revision,
            },
        )

    def _create_modeling_geometry(
        self,
        definition: ProfileDefinition | PathDefinition | dict[str, Any],
        kind: str,
        expected_revision: str | None,
    ) -> dict[str, Any]:
        try:
            request = (
                modeling_request_from_dict(definition)
                if isinstance(definition, dict)
                else definition
            )
            if request.kind != kind:
                raise ValueError(f"Expected a {kind} definition.")
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return self._error(f"Invalid modeling request: {exc}", error_code="invalid_cad_request")
        return self.apply_modeling(request, expected_revision=expected_revision or "")

    def update_profile(
        self,
        profile_id: str,
        workplane_id: str | None = None,
        expected_revision: str | None = None,
        *,
        definition: ProfileDefinition | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update the same native sketch from typed geometry or a workplane."""
        return self._profile_operation(
            "update_profile",
            {
                "profile_id": profile_id,
                "workplane_id": workplane_id,
                "expected_revision": expected_revision,
                **(
                    {
                        "definition": definition.to_dict()
                        if hasattr(definition, "to_dict")
                        else definition
                    }
                    if definition is not None
                    else {}
                ),
            },
        )

    def update_feature(
        self,
        feature_id: str,
        parameters: LoftFeatureUpdate | SweepFeatureUpdate | dict[str, Any],
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Edit declared native loft/sweep parameters without replacing a feature."""
        return self._profile_operation(
            "update_feature",
            {
                "feature_id": feature_id,
                "parameters": parameters.to_dict()
                if hasattr(parameters, "to_dict")
                else parameters,
                "expected_revision": expected_revision,
            },
        )

    def inspect_capabilities(self) -> dict[str, Any]:
        """Inspect the selected integration's supported operations and restrictions."""
        if self._live_backend is not None:
            inspector = getattr(self._live_backend.capabilities, "inspect_capabilities", None)
            if not callable(inspector):
                return self._operation_error(
                    "Selected integration does not expose capability inspection.",
                    error_code="cad_operation_not_supported",
                )
            return inspector()
        return self._profile_operation("inspect_capabilities", {})

    def inspect_geometry(self, object_ids: list[str] | None = None) -> dict[str, Any]:
        """Report native validity, dimensions and dependencies without handles."""
        return self._profile_operation("inspect_geometry", {"object_ids": object_ids})

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
        return self._profile_operation(
            "loft",
            {
                "profile_workplane_ids": profile_workplane_ids,
                "profile_ids": profile_ids,
                "make_solid": make_solid,
                "ruled": ruled,
                **({"alignment": alignment} if alignment != "automatic" else {}),
                "expected_revision": expected_revision,
            },
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
        return self._profile_operation(
            "sweep",
            {
                "profile_id": profile_id,
                "path_id": path_id,
                "make_solid": make_solid,
                "orientation": orientation,
                "transition": transition,
                "expected_revision": expected_revision,
            },
        )

    def update_path(
        self,
        path_id: str,
        definition: PathDefinition | dict[str, Any],
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Replace an unconstrained planar path in place and recompute dependents."""
        return self._profile_operation(
            "update_path",
            {
                "path_id": path_id,
                "definition": definition.to_dict()
                if hasattr(definition, "to_dict")
                else definition,
                "expected_revision": expected_revision,
            },
        )

    def cut(self, target_id: str, tool_id: str) -> Dict[str, Any]:
        ready = self._ensure_gui_session()
        if ready is not None:
            return ready
        if self._worker is not None:
            return self._worker.call("cut", {"target_id": target_id, "tool_id": tool_id})
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
                "union",
                {
                    "target_id": target_id,
                    "tool_ids": tool_ids,
                    "expected_revision": expected_revision,
                },
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
        self._recalculate_document_revision()
        return self._operation_ok(
            summary=f"United {target_id} with {len(tool_ids)} shape(s)",
            document_revision=self.document_revision,
            changed_object_ids=[target_id],
            support=CadOperationSupport(
                operation="union",
                status="supported",
                mode="native_dependent_feature",
                reason="Created a native boolean feature that references its operands.",
            ),
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
                termination=(HoleTermination.THROUGH if through else HoleTermination.BLIND),
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
        target = (
            candidate if isinstance(candidate, CadObject) else getattr(candidate, "feature", None)
        )
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

    def export_step(self, path: str, shape_id: Optional[str] = None) -> Dict[str, Any]:
        return self._export(path, "step", shape_id=shape_id)

    def export_stl(self, path: str, shape_id: Optional[str] = None) -> Dict[str, Any]:
        return self._export(path, "stl", shape_id=shape_id)

    def export_native(self, path: str, shape_id: Optional[str] = None) -> Dict[str, Any]:
        return self._export(path, "native", shape_id=shape_id)

    def _export(self, path: str, kind: str, shape_id: Optional[str] = None) -> Dict[str, Any]:
        if self._worker is not None:
            return self._worker.call(f"export_{kind}", {"path": path, "shape_id": shape_id})
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
                    return self._error(f"Object '{shape_id}' has no geometry export wrapper.")
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
