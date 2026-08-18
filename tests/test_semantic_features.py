from __future__ import annotations

from dataclasses import dataclass

import pytest

from rapidcadpy.cad_objects import CadDocument, CadFeature
from rapidcadpy.dimension_planner import ModelBounds, build_dimension_intents_from_features
from rapidcadpy.feature_executor import FeatureResult, FeatureSupport
from rapidcadpy.features import (
    FeatureProvenance,
    FilletFeature,
    GeometrySelection,
    HoleFeature,
    HoleTermination,
    feature_from_dict,
)
from rapidcadpy.integrations.freecad.drawing import FreeCADDrawingBackend


def test_semantic_feature_round_trip_has_no_native_handle() -> None:
    feature = FilletFeature(
        target_id="body-1",
        edges=(GeometrySelection("body-1", ("Edge2", "Edge5"), "sha256:one"),),
        radius_mm=3.0,
        reference_point=(1.0, 2.0, 3.0),
    )

    payload = feature.to_dict()
    restored = feature_from_dict(payload)

    assert isinstance(restored, FilletFeature)
    assert restored.edges[0].subelements == ("Edge2", "Edge5")
    assert "native_handle" not in str(payload)


def test_blind_hole_requires_depth_and_through_hole_rejects_one() -> None:
    with pytest.raises(ValueError, match="positive depth"):
        HoleFeature(
            target_id="body-1",
            diameter_mm=5.0,
            termination=HoleTermination.BLIND,
        )
    with pytest.raises(ValueError, match="must not declare depth"):
        HoleFeature(
            target_id="body-1",
            diameter_mm=5.0,
            termination=HoleTermination.THROUGH,
            depth_mm=10.0,
        )


def test_semantic_dimension_plan_uses_hole_and_fillet_intent() -> None:
    holes = tuple(
        HoleFeature(
            target_id="body-1",
            center=(x, y, 5.0),
            diameter_mm=10.0,
            termination=HoleTermination.THROUGH,
        )
        for x, y in ((20.0, 15.0), (60.0, 15.0), (20.0, 35.0), (60.0, 35.0))
    )
    fillet = FilletFeature(
        target_id="body-1",
        edges=(GeometrySelection("body-1", ("Edge1",)),),
        radius_mm=3.0,
        reference_point=(0.0, 0.0, 5.0),
        provenance=FeatureProvenance.NATIVE_HYDRATED,
    )

    intents = build_dimension_intents_from_features(
        ModelBounds(0.0, 80.0, 0.0, 50.0, 0.0, 10.0),
        (*holes, fillet),
    )

    hole = next(item for item in intents if item.kind == "hole")
    assert hole.formatted_text("ISO") == "4X ⌀10,00 THRU"
    assert {item.value_mm for item in intents if item.kind == "spacing"} == {20.0, 40.0}
    assert next(item for item in intents if item.kind == "radius").formatted_text("ISO") == "R3,00"


@dataclass
class _Executor:
    result: FeatureResult
    received: tuple[object, object, object] | None = None

    def inspect_support(self, definition, target):  # type: ignore[no-untyped-def]
        return FeatureSupport("fillet", "supported", "native", "test")

    def apply(self, definition, target, expected_revision):  # type: ignore[no-untyped-def]
        self.received = (definition, target, expected_revision)
        return self.result


def test_feature_apply_delegates_to_executor() -> None:
    document = CadDocument(backend="test", native_handle=object(), adapter=None)  # type: ignore[arg-type]
    target = CadFeature("body-1", document, native_handle=object())
    feature = FilletFeature(
        target_id="body-1", edges=(GeometrySelection("body-1", ("Edge1",)),)
    )
    executor = _Executor(FeatureResult(target, feature, "sha256:new"))

    result = feature.apply(executor, target, "sha256:old")

    assert result.document_revision == "sha256:new"
    assert executor.received == (feature, target, "sha256:old")


def test_freecad_drawing_prefers_persisted_semantic_features() -> None:
    document = CadDocument(backend="freecad", native_handle=object(), adapter=None)  # type: ignore[arg-type]
    source_id = document.object_id("ExternalObject")
    feature = FilletFeature(
        target_id=source_id,
        edges=(GeometrySelection(source_id, ("Edge1",)),),
        radius_mm=2.0,
        reference_point=(0.0, 0.0, 0.0),
    )
    document.register_feature_definition("RapidCADFillet1", feature.to_dict())

    source = type("Source", (), {"Name": "ExternalObject", "Base": None})()
    restored = FreeCADDrawingBackend._semantic_features_for_sources(document, [source])

    assert len(restored) == 1
    assert isinstance(restored[0], FilletFeature)
