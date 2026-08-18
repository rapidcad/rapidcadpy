"""Native, history-preserving execution of semantic RapidCADPy features."""

from __future__ import annotations

import json
from typing import Any, Optional

from ...cad_objects import CadFeature, CadObject
from ...feature import Feature
from ...feature_executor import FeatureExecutor, FeatureResult, FeatureSupport
from ...features import (
    ChamferFeature,
    FilletFeature,
    GeometrySelection,
    HoleFeature,
    HoleTermination,
    HoleType,
)
from .errors import FreeCADNativeFeatureError


class FreeCADFeatureExecutor(FeatureExecutor):
    """Apply semantic definitions as dependent FreeCAD document features."""

    def inspect_support(self, definition: Feature, target: CadObject) -> FeatureSupport:
        if target.document.backend != "freecad":
            return FeatureSupport(
                operation=type(definition).__name__,
                status="unsupported",
                mode="none",
                reason="Target is not owned by a FreeCAD document.",
            )
        if getattr(target.native_handle, "Shape", None) is None:
            return FeatureSupport(
                operation=type(definition).__name__,
                status="unsupported",
                mode="none",
                reason="Target has no native FreeCAD shape.",
            )
        if isinstance(definition, FilletFeature):
            return FeatureSupport(
                operation="fillet",
                status="supported",
                mode="native_dependent_feature",
                reason="Creates a dependent Part::Fillet referencing the target.",
            )
        if isinstance(definition, ChamferFeature):
            return FeatureSupport(
                operation="chamfer",
                status="supported",
                mode="native_dependent_feature",
                reason="Creates a dependent Part::Chamfer referencing the target.",
            )
        if isinstance(definition, HoleFeature):
            if definition.hole_type not in {HoleType.SIMPLE, HoleType.COUNTERSINK}:
                return FeatureSupport(
                    operation="hole", status="conditional", mode="native_dependent_feature",
                    reason="Counterbore holes are not enabled by the FreeCAD executor.",
                )
            return FeatureSupport(
                operation="hole",
                status="supported",
                mode="native_dependent_feature",
                reason="Creates a parametric cylinder and dependent Part::Cut.",
            )
        return FeatureSupport(
            operation=type(definition).__name__,
            status="unsupported",
            mode="none",
            reason="No FreeCAD executor is registered for this feature definition.",
        )

    def apply(
        self,
        definition: Feature,
        target: CadObject,
        expected_revision: Optional[str],
    ) -> FeatureResult:
        support = self.inspect_support(definition, target)
        if support.status != "supported":
            raise FreeCADNativeFeatureError(
                f"{support.operation} is {support.status}: {support.reason}"
            )
        current_revision = target.document.revision
        if expected_revision and current_revision and expected_revision != current_revision:
            raise FreeCADNativeFeatureError(
                "Document revision mismatch: expected "
                f"{expected_revision}, current {current_revision}."
            )
        if isinstance(definition, FilletFeature):
            native_feature = self._apply_fillet(definition, target)
            auxiliary_features: tuple[Any, ...] = ()
        elif isinstance(definition, ChamferFeature):
            native_feature = self._apply_chamfer(definition, target)
            auxiliary_features = ()
        elif isinstance(definition, HoleFeature):
            native_feature, auxiliary_features = self._apply_hole(definition, target)
        else:  # safeguarded by inspect_support
            raise FreeCADNativeFeatureError("Unsupported FreeCAD feature definition.")
        wrapped = self._wrap_feature(target, native_feature)
        self._persist_definition(native_feature, definition)
        target.document.register_feature_definition(native_feature.Name, definition.to_dict())
        return FeatureResult(
            feature=wrapped,
            definition=definition,
            document_revision=target.document.revision,
            created_object_ids=(
                *(target.document.object_id(item.Name) for item in auxiliary_features),
                wrapped.id,
            ),
            changed_object_ids=(target.id,),
        )

    def _apply_fillet(self, definition: FilletFeature, target: CadObject) -> Any:
        edge_indexes = self._resolve_edge_indexes(definition.edges, target)
        doc = target.document.native_handle
        native = None
        try:
            native = doc.addObject("Part::Fillet", self._next_name(doc, "RapidCADFillet"))
            native.Base = target.native_handle
            native.Edges = [
                (edge_index, float(definition.radius_mm), float(definition.radius_mm))
                for edge_index in edge_indexes
            ]
            native.EdgeLinks = (
                target.native_handle,
                [f"Edge{edge_index}" for edge_index in edge_indexes],
            )
            target.document.recompute()
            self._validate_result(native, "Part::Fillet")
            return native
        except Exception as exc:
            self._remove_failed_feature(doc, native)
            raise FreeCADNativeFeatureError(
                "Could not create native Part::Fillet; no direct-geometry fallback "
                "was applied."
            ) from exc

    def _apply_chamfer(self, definition: ChamferFeature, target: CadObject) -> Any:
        edge_indexes = self._resolve_edge_indexes(definition.edges, target)
        doc = target.document.native_handle
        native = None
        try:
            native = doc.addObject("Part::Chamfer", self._next_name(doc, "RapidCADChamfer"))
            native.Base = target.native_handle
            native.Edges = [
                (edge_index, float(definition.distance_mm), float(definition.distance_mm))
                for edge_index in edge_indexes
            ]
            target.document.recompute()
            self._validate_result(native, "Part::Chamfer")
            return native
        except Exception as exc:
            self._remove_failed_feature(doc, native)
            raise FreeCADNativeFeatureError(
                "Could not create native Part::Chamfer; no direct-geometry fallback "
                "was applied."
            ) from exc

    def _apply_hole(
        self,
        definition: HoleFeature,
        target: CadObject,
    ) -> tuple[Any, tuple[Any, ...]]:
        """Create native simple or countersunk tooling and a dependent cut."""

        import FreeCAD as App

        doc = target.document.native_handle
        cylinder = cone = fuse = cut = None
        try:
            axis = self._normalized_axis(definition.axis)
            if definition.termination is HoleTermination.THROUGH:
                bounds = target.native_handle.Shape.BoundBox
                depth = float(bounds.DiagonalLength) + 2.0 * definition.diameter_mm
                start = tuple(
                    coordinate - direction * definition.diameter_mm
                    for coordinate, direction in zip(definition.center, axis)
                )
            else:
                assert definition.depth_mm is not None
                depth = float(definition.depth_mm)
                start = definition.center
            cylinder = doc.addObject(
                "Part::Cylinder", self._next_name(doc, "RapidCADHoleTool")
            )
            cylinder.Radius = float(definition.diameter_mm) / 2.0
            cylinder.Height = depth
            cylinder.Placement = App.Placement(
                App.Vector(*start),
                App.Rotation(App.Vector(0.0, 0.0, 1.0), App.Vector(*axis)),
            )
            tool = cylinder
            auxiliary: tuple[Any, ...] = (cylinder,)
            if definition.hole_type is HoleType.COUNTERSINK:
                assert definition.countersink_diameter_mm is not None
                assert definition.countersink_angle_degrees is not None
                radius_delta = (
                    definition.countersink_diameter_mm - definition.diameter_mm
                ) / 2.0
                import math

                cone_depth = radius_delta / math.tan(
                    math.radians(definition.countersink_angle_degrees / 2.0)
                )
                cone = doc.addObject(
                    "Part::Cone", self._next_name(doc, "RapidCADCountersinkTool")
                )
                cone.Radius1 = float(definition.countersink_diameter_mm) / 2.0
                cone.Radius2 = float(definition.diameter_mm) / 2.0
                cone.Height = cone_depth
                cone.Placement = App.Placement(
                    App.Vector(*definition.center),
                    App.Rotation(App.Vector(0.0, 0.0, 1.0), App.Vector(*axis)),
                )
                fuse = doc.addObject("Part::MultiFuse", self._next_name(doc, "RapidCADHoleTool"))
                fuse.Shapes = [cylinder, cone]
                tool = fuse
                auxiliary = (cylinder, cone, fuse)
            cut = doc.addObject("Part::Cut", self._next_name(doc, "RapidCADHole"))
            cut.Base = target.native_handle
            cut.Tool = tool
            doc.recompute()
            self._validate_result(cut, "Part::Cut")
            target.document.adapter.set_boolean_result_visibility(
                cut,
                [target.native_handle, *auxiliary],
            )
            return cut, auxiliary
        except Exception as exc:
            self._remove_failed_feature(doc, cut)
            self._remove_failed_feature(doc, fuse)
            self._remove_failed_feature(doc, cone)
            self._remove_failed_feature(doc, cylinder)
            raise FreeCADNativeFeatureError(
                "Could not create a native simple-hole feature; no direct-geometry "
                "fallback was applied."
            ) from exc

    def _resolve_edge_indexes(
        self,
        selections: tuple[GeometrySelection, ...],
        target: CadObject,
    ) -> list[int]:
        indexes: list[int] = []
        for selection in selections:
            if selection.object_id != target.id:
                raise FreeCADNativeFeatureError("Selection does not belong to target.")
            if (
                selection.document_revision
                and target.document.revision
                and selection.document_revision != target.document.revision
            ):
                raise FreeCADNativeFeatureError(
                    "Selection revision is stale; refresh the object and select edges again."
                )
            for subelement in selection.subelements:
                if not subelement.startswith("Edge") or not subelement[4:].isdigit():
                    raise FreeCADNativeFeatureError(
                        f"Unsupported FreeCAD edge selection {subelement!r}."
                    )
                index = int(subelement[4:])
                edges = list(getattr(target.native_handle.Shape, "Edges", []))
                if index < 1 or index > len(edges):
                    raise FreeCADNativeFeatureError(
                        f"Selected edge {subelement!r} is not present on target."
                    )
                indexes.append(index)
        return sorted(set(indexes))

    @staticmethod
    def _normalized_axis(axis: tuple[float, float, float]) -> tuple[float, float, float]:
        length = sum(component * component for component in axis) ** 0.5
        if length <= 1.0e-9:
            raise FreeCADNativeFeatureError("Hole axis must be non-zero.")
        return tuple(component / length for component in axis)  # type: ignore[return-value]

    @staticmethod
    def _next_name(document: Any, prefix: str) -> str:
        index = 1
        while document.getObject(f"{prefix}{index}") is not None:
            index += 1
        return f"{prefix}{index}"

    @staticmethod
    def _validate_result(native_feature: Any, type_id: str) -> None:
        result = getattr(native_feature, "Shape", None)
        if result is None or result.isNull():
            raise ValueError(f"FreeCAD recomputed an empty {type_id}.")

    @staticmethod
    def _remove_failed_feature(document: Any, native_feature: Any) -> None:
        if native_feature is None:
            return
        try:
            document.removeObject(native_feature.Name)
            document.recompute()
        except Exception:
            pass

    @staticmethod
    def _wrap_feature(target: CadObject, native_feature: Any) -> CadFeature:
        document = target.document
        native_name = str(native_feature.Name)
        return CadFeature(
            id=document.object_id(native_name),
            document=document,
            native_handle=native_feature,
            native_name=native_name,
            native_type=str(native_feature.TypeId),
            label=str(getattr(native_feature, "Label", native_name)),
            capabilities=frozenset({"geometry", "get_properties", "set_properties", "feature_history"}),
        )

    @staticmethod
    def _persist_definition(native_feature: Any, definition: Feature) -> None:
        for name, value in (
            ("RapidCADFeatureId", str(definition.id)),
            ("RapidCADFeatureKind", definition.to_dict()["kind"]),
            ("RapidCADFeatureSchemaVersion", "1"),
            ("RapidCADFeatureDefinition", json.dumps(definition.to_dict(), sort_keys=True)),
        ):
            if not hasattr(native_feature, name):
                native_feature.addProperty("App::PropertyString", name, "RapidCAD")
            setattr(native_feature, name, value)
