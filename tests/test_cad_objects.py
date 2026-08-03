"""Tests for backend-neutral references to live native CAD objects."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rapidcadpy import CadDocument, CadFeature
from rapidcadpy.integrations.freecad.shape import FreeCADShape


class FakeAdapter:
    backend_name = "fake"

    def __init__(self):
        self.recompute_count = 0
        self.saved_paths = []

    def recompute(self, document):
        self.recompute_count += 1
        document.recomputed = True

    def save_document(self, document, path=None):
        saved_path = path or document.file_name
        self.saved_paths.append(saved_path)
        return saved_path

    def get_property(self, obj, name):
        return obj.properties[name]

    def set_property(self, obj, name, value):
        obj.properties[name] = value

    def get_shape(self, obj):
        return obj.Shape


def test_cad_feature_delegates_to_live_native_handle():
    adapter = FakeAdapter()
    native_document = SimpleNamespace(file_name="/tmp/model.native")
    native_shape = object()
    native_feature = SimpleNamespace(
        properties={"Length": 10.0},
        Shape=native_shape,
    )
    document = CadDocument(
        backend="fake",
        native_handle=native_document,
        adapter=adapter,
        name="model",
        revision="sha256:before-edit",
    )
    feature = CadFeature(
        id="native_1",
        document=document,
        native_handle=native_feature,
        native_name="Extrude1",
        capabilities=frozenset(
            {
                "geometry",
                "get_properties",
                "set_properties",
                "feature_history",
            }
        ),
    )

    feature.set_property("Length", 25.0)

    assert feature.get_property("Length") == 25.0
    assert native_feature.properties["Length"] == 25.0
    assert feature.shape is native_shape
    assert feature.is_parametric is True
    assert document.revision is None
    assert adapter.recompute_count == 1
    assert "native_handle" not in feature.to_dict()


def test_cad_object_rejects_unadvertised_property_mutation():
    adapter = FakeAdapter()
    document = CadDocument(
        backend="fake",
        native_handle=SimpleNamespace(),
        adapter=adapter,
    )
    feature = CadFeature(
        id="native_1",
        document=document,
        native_handle=SimpleNamespace(),
    )

    assert feature.is_parametric is False
    with pytest.raises(NotImplementedError, match="set_properties"):
        feature.set_property("Length", 25.0)


def test_cad_document_owns_stable_one_to_one_native_object_ids():
    document = CadDocument(
        backend="fake",
        native_handle=SimpleNamespace(),
        adapter=FakeAdapter(),
    )
    generated_ids = iter(["native_1", "native_2", "native_3"]).__next__

    pad_id = document.object_id("Pad", generated_ids)
    inserted_id = document.object_id("InsertedBeforePad", generated_ids)

    assert pad_id == "native_1"
    assert inserted_id == "native_2"
    assert document.object_id("Pad", generated_ids) == pad_id

    document.bind_object_id("Fillet", pad_id)

    assert document.object_id("Fillet", generated_ids) == pad_id
    assert document.object_id("Pad", generated_ids) == "native_3"


def test_freecad_shape_exposes_backend_neutral_document_and_feature():
    native_document = SimpleNamespace(
        Name="Doc",
        Label="Document",
        FileName="",
    )
    native_feature = SimpleNamespace(
        Name="Pad",
        Label="Pad",
        TypeId="Part::Extrusion",
    )
    cad_document = CadDocument(
        backend="freecad",
        native_handle=native_document,
        adapter=FakeAdapter(),
        name="Doc",
    )

    class FakeApp:
        def __init__(self):
            self.registered = []

        def get_cad_document(self):
            return cad_document

        def register_shape(self, shape):
            self.registered.append(shape)

    app = FakeApp()
    shape = FreeCADShape(
        obj=object(),
        app=app,
        doc=native_document,
        current_feature=native_feature,
    )

    assert shape.document is cad_document
    assert shape.feature.native_handle is native_feature
    assert shape.feature.id == "native_1"
    assert shape.is_parametric is True
    assert shape._doc is native_document
    assert shape._current_feature is native_feature
    assert app.registered == [shape]

    replacement_feature = SimpleNamespace(
        Name="Fillet",
        Label="Fillet",
        TypeId="PartDesign::Fillet",
    )
    original_id = shape.feature.id
    shape._bind_feature(replacement_feature)

    assert shape.feature.id == original_id
    assert cad_document.object_id("Fillet") == original_id
    assert cad_document.object_id("Pad") != original_id
