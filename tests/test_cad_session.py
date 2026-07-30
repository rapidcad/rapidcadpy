"""Tests for RapidCADPy's standalone stateful CAD API."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from rapidcadpy.cad_objects import CadDocument, CadFeature
from rapidcadpy.cad_session import CadSession
from rapidcadpy.drawing import DrawingResult


def test_session_reports_missing_backend_before_setup():
    session = CadSession()

    result = session.rect(10, 20)

    assert result["ok"] is False
    assert "setup_backend" in result["error"]


def test_embedded_session_can_create_first_gui_document(monkeypatch):
    document = SimpleNamespace(
        Name="Demo",
        Label="Demo",
        FileName="",
        Objects=[],
        recompute=lambda: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "FreeCAD",
        SimpleNamespace(newDocument=lambda name: document),
    )
    session = CadSession(execution_mode="embedded")

    result = session.new_document("Demo")

    assert result["ok"] is True
    assert result["summary"] == "Created document 'Demo'"
    assert session.backend_name == "freecad"
    assert session.app.get_doc() is document


def test_open_document_hydrates_native_objects_and_keeps_handles(
    tmp_path,
    monkeypatch,
):
    class FakeShape:
        Volume = 120.0
        Area = 52.0
        Faces = [1, 2, 3, 4, 5, 6]
        Edges = list(range(12))
        Vertexes = list(range(8))
        BoundBox = SimpleNamespace(
            XMin=0,
            XMax=10,
            YMin=0,
            YMax=4,
            ZMin=0,
            ZMax=3,
            XLength=10,
            YLength=4,
            ZLength=3,
        )

        def isNull(self):
            return False

    class FakeObject:
        def __init__(self, name, type_id, properties):
            self.Name = name
            self.Label = name
            self.TypeId = type_id
            self.PropertiesList = list(properties)
            self._properties = properties
            for property_name, value in properties.items():
                setattr(self, property_name, value)
            self.InList = []
            self.OutList = []
            self.Group = []
            self.ViewObject = SimpleNamespace(Visibility=True)

        def getPropertyByName(self, name):
            return getattr(self, name)

        def getParentGeoFeatureGroup(self):
            return None

    sketch = FakeObject("Sketch", "Sketcher::SketchObject", {"Label": "Sketch"})
    pad = FakeObject(
        "Pad",
        "Part::Extrusion",
        {"LengthFwd": 10.0, "Shape": FakeShape()},
    )
    body = FakeObject("Body", "PartDesign::Body", {"Group": [sketch, pad]})
    body.Group = [sketch, pad]
    pad.OutList = [sketch]
    sketch.InList = [pad]
    document = SimpleNamespace(
        Name="Bracket",
        Label="Bracket",
        FileName="",
        Objects=[body, sketch, pad],
    )
    document.recompute = lambda: None
    document_path = tmp_path / "bracket.FCStd"
    document_path.touch()

    scratch = SimpleNamespace(Name="Scratch", FileName="")
    fake_app = SimpleNamespace(
        get_doc=lambda: scratch,
        _fc_doc=scratch,
        _feature_counter=0,
        _shapes=[],
        _workplanes=[],
    )
    monkeypatch.setitem(
        sys.modules,
        "FreeCAD",
        SimpleNamespace(
            closeDocument=lambda name: None,
            openDocument=lambda path: document,
        ),
    )
    session = CadSession()
    session.backend_name = "freecad"
    session.app = fake_app

    result = session.open_document(str(document_path))

    assert result["ok"] is True
    assert result["object_count"] == 3
    assert result["document_revision"].startswith("sha256:")
    ids = {item["native_name"]: item["id"] for item in result["objects"]}
    assert session.runtime_objects[ids["Pad"]].native_handle is pad
    assert session.runtime_objects[ids["Pad"]].document.native_handle is document
    assert session.runtime_objects[ids["Pad"]].is_parametric is True
    assert next(item for item in result["objects"] if item["native_name"] == "Body")[
        "properties"
    ]["Group"] == [{"object_id": ids["Sketch"]}, {"object_id": ids["Pad"]}]
    assert next(item for item in result["objects"] if item["native_name"] == "Pad")[
        "dependencies"
    ]["depends_on"] == [ids["Sketch"]]
    assert next(item for item in result["objects"] if item["native_name"] == "Pad")[
        "geometry"
    ]["volume"] == pytest.approx(120.0)
    assert result["tree"][0]["id"] == ids["Body"]
    assert [item["id"] for item in result["tree"][0]["children"]] == [
        ids["Sketch"],
        ids["Pad"],
    ]

    changed = session.set_object_property(
        ids["Pad"],
        "LengthFwd",
        25.0,
        expected_revision=result["document_revision"],
    )

    assert changed["ok"] is True
    assert pad.LengthFwd == 25.0
    assert changed["value"] == 25.0
    assert changed["document_revision"] != result["document_revision"]

    stale_change = session.set_object_property(
        ids["Pad"],
        "LengthFwd",
        30.0,
        expected_revision=result["document_revision"],
    )

    assert stale_change["ok"] is False
    assert "revision mismatch" in stale_change["error"]
    assert pad.LengthFwd == 25.0


def test_extrusion_id_survives_parameter_rehydration_and_object_insertion():
    class FakeAdapter:
        backend_name = "freecad"
        parameter_adapter = None

        def recompute(self, document):
            document.recompute()

        def save_document(self, document, path=None):
            return path or document.FileName

        def get_property(self, obj, name):
            return obj.getPropertyByName(name)

        def set_property(self, obj, name, value):
            setattr(obj, name, value)

        def get_shape(self, obj):
            return getattr(obj, "Shape", None)

    class FakeNativeObject:
        def __init__(self, name, type_id, shape=None):
            self.Name = name
            self.Label = name
            self.TypeId = type_id
            self.Shape = shape
            self.PropertiesList = ["Shape"] if shape is not None else []
            self.InList = []
            self.OutList = []
            self.Group = []
            self.ViewObject = SimpleNamespace(Visibility=True)

        def getPropertyByName(self, name):
            return getattr(self, name)

        def getParentGeoFeatureGroup(self):
            return None

    native_shape = SimpleNamespace(
        Volume=100.0,
        Area=70.0,
        Faces=[1, 2, 3, 4, 5, 6],
        Edges=list(range(12)),
        Vertexes=list(range(8)),
        isNull=lambda: False,
    )
    extrusion = FakeNativeObject("Extrude", "Part::Extrusion", native_shape)
    native_document = SimpleNamespace(
        Name="StableIds",
        Label="Stable IDs",
        FileName="",
        Objects=[extrusion],
        recompute=lambda: None,
    )
    cad_document = CadDocument(
        backend="freecad",
        native_handle=native_document,
        adapter=FakeAdapter(),
        name=native_document.Name,
    )
    feature = CadFeature(
        id=cad_document.object_id("Extrude"),
        document=cad_document,
        native_handle=extrusion,
        native_name="Extrude",
        native_type="Part::Extrusion",
        label="Extrude",
        capabilities=frozenset({"geometry", "feature_history"}),
    )
    runtime_shape = SimpleNamespace(obj=native_shape, feature=feature)
    workplane = SimpleNamespace(extrude=lambda *args, **kwargs: runtime_shape)

    session = CadSession()
    session.backend_name = "freecad"
    session.app = SimpleNamespace(cad_document=cad_document)
    session.cad_document = cad_document
    session.active_workplane = workplane
    session.active_workplane_id = "workplane_1"

    created = session.extrude(10)
    extrusion_id = created["object_id"]

    inserted = FakeNativeObject("InsertedBeforeExtrude", "Part::Feature")
    native_document.Objects.insert(0, inserted)
    rehydrated = session._hydrate_freecad_document(
        native_document,
        None,
        source_tool="create_parameter",
    )
    inspected = session.get_object(extrusion_id)
    ids_by_name = {item["native_name"]: item["id"] for item in rehydrated["objects"]}

    assert extrusion_id == "shape_1"
    assert ids_by_name["Extrude"] == extrusion_id
    assert ids_by_name["InsertedBeforeExtrude"] != extrusion_id
    assert inspected["ok"] is True
    assert inspected["object"]["native_name"] == "Extrude"


def test_generate_drawing_uses_backend_factory_and_rehydrates_document(
    tmp_path,
    monkeypatch,
):
    class FakeAdapter:
        backend_name = "freecad"
        parameter_adapter = None

        def recompute(self, document):
            document.recompute()

        def save_document(self, document, path=None):
            return path or document.FileName

        def get_property(self, obj, name):
            return getattr(obj, name)

        def set_property(self, obj, name, value):
            setattr(obj, name, value)

        def get_shape(self, obj):
            return getattr(obj, "Shape", None)

    shape = SimpleNamespace(
        Volume=100.0,
        Area=70.0,
        Faces=[1],
        Edges=[1],
        Vertexes=[1],
        BoundBox=SimpleNamespace(
            XMin=0,
            XMax=10,
            YMin=0,
            YMax=5,
            ZMin=0,
            ZMax=2,
            XLength=10,
            YLength=5,
            ZLength=2,
        ),
        isNull=lambda: False,
    )
    source = SimpleNamespace(
        Name="Pad",
        Label="Pad",
        TypeId="PartDesign::Feature",
        Shape=shape,
        PropertiesList=["Shape"],
        InList=[],
        OutList=[],
        Group=[],
        ViewObject=SimpleNamespace(Visibility=True),
        getPropertyByName=lambda name: shape if name == "Shape" else None,
        getParentGeoFeatureGroup=lambda: None,
    )
    native_document = SimpleNamespace(
        Name="Bracket",
        Label="Bracket",
        FileName="",
        Objects=[source],
        recompute=lambda: None,
    )
    source.Document = native_document
    cad_document = CadDocument(
        backend="freecad",
        native_handle=native_document,
        adapter=FakeAdapter(),
        name="Bracket",
        label="Bracket",
    )
    object_id = cad_document.object_id("Pad", lambda: "shape_1")
    feature = CadFeature(
        id=object_id,
        document=cad_document,
        native_handle=source,
        native_name="Pad",
        native_type=source.TypeId,
        label=source.Label,
        capabilities=frozenset({"geometry", "feature_history"}),
    )
    calls = []

    class FakeDrawingBackend:
        def generate_drawing(self, **kwargs):
            calls.append(kwargs)
            pdf = kwargs["output_directory"] / "Bracket_run-1_2026-07-24.pdf"
            pdf.write_bytes(b"%PDF drawing")
            return DrawingResult(
                pdf_path=pdf,
                page_name="RapidCADDrawing",
                source_object_ids=(object_id,),
            )

    monkeypatch.setattr(
        "rapidcadpy.drawing.create_drawing_backend",
        lambda document: FakeDrawingBackend(),
    )
    session = CadSession(execution_mode="embedded")
    session.backend_name = "freecad"
    session.app = SimpleNamespace(cad_document=cad_document)
    session.cad_document = cad_document
    session.runtime_objects[object_id] = feature
    session.document_revision = "sha256:model"

    result = session.generate_drawing(
        object_ids=[object_id],
        output_directory=str(tmp_path),
        part_name="Bracket",
        run_id="run-1",
        expected_revision="sha256:model",
    )

    assert result["ok"] is True
    assert result["pdf_path"].endswith(".pdf")
    assert result["document_revision"].startswith("sha256:")
    assert calls[0]["document"] is cad_document
    assert calls[0]["objects"] == [feature]
    assert calls[0]["include_native"] is True
