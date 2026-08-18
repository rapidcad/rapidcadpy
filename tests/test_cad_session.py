"""Tests for RapidCADPy's standalone stateful CAD API."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from rapidcadpy.cad_objects import CadDocument, CadFeature
from rapidcadpy.cad_session import CadSession, SemanticObject
from rapidcadpy.drawing import DrawingResult
from rapidcadpy.features import HoleFeature, HoleType


def test_session_reports_missing_backend_before_setup():
    session = CadSession()

    result = session.rect(10, 20)

    assert result["ok"] is False
    assert "setup_backend" in result["error"]


def test_list_objects_is_concise_and_get_object_requires_the_exact_id():
    session = CadSession()
    session.objects = {
        "native_1": SemanticObject(
            id="native_1",
            type="sketch",
            label="Sketch_1",
            source_op="op_1",
            metadata={
                "native_name": "Sketch_1",
                "properties": {"Geometry": ["large payload"]},
                "geometry": {"edge_count": 4},
            },
        ),
        "shape_1": SemanticObject(
            id="shape_1",
            type="feature",
            label="Extrude_1",
            source_op="op_1",
            metadata={
                "native_name": "Extrude_1",
                "properties": {"LengthFwd": 60},
                "geometry": {"volume": 216000},
            },
        ),
    }

    listed = session.list_objects()

    assert listed["ok"] is True
    assert listed["objects"] == [
        {
            "id": "native_1",
            "type": "sketch",
            "label": "Sketch_1",
            "native_name": "Sketch_1",
            "geometry": {"edge_count": 4},
        },
        {
            "id": "shape_1",
            "type": "feature",
            "label": "Extrude_1",
            "native_name": "Extrude_1",
            "geometry": {"volume": 216000},
        },
    ]
    assert "complete id value" in listed["id_usage"]
    assert "properties" not in listed["objects"][0]

    inspected = session.get_object("shape_1")

    assert inspected["ok"] is True
    assert inspected["object"]["properties"] == {"LengthFwd": 60}

    invalid = session.get_object("1")

    assert invalid["ok"] is False
    assert "include their prefix" in invalid["error"]
    assert "'native_1', 'shape_1'" in invalid["error"]


def test_get_object_exposes_geometry_driven_hole_dimension_requests():
    native_source = SimpleNamespace(Name="Pad", Base=None)
    native_hole = SimpleNamespace(Name="RapidCADHole", Base=native_source)
    document = CadDocument(
        backend="freecad",
        native_handle=SimpleNamespace(),
        adapter=SimpleNamespace(),
    )
    document.bind_object_id("Pad", "shape_1")
    document.bind_object_id("RapidCADHole", "shape_2")
    definition = HoleFeature(
        target_id="shape_1",
        center=(0.0, 0.0, 60.0),
        diameter_mm=8.0,
        hole_type=HoleType.COUNTERSINK,
        countersink_diameter_mm=16.0,
        countersink_angle_degrees=90.0,
    ).to_dict()
    document.register_feature_definition("RapidCADHole", definition)
    runtime = CadFeature(
        id="shape_2",
        document=document,
        native_handle=native_hole,
        native_name="RapidCADHole",
    )
    session = CadSession(execution_mode="embedded")
    session.cad_document = document
    session.runtime_objects["shape_2"] = runtime
    session.objects["shape_2"] = SemanticObject(
        id="shape_2",
        type="feature",
        label="RapidCADHole",
        source_op="op_1",
    )

    result = session.get_object("shape_2")

    assert result["ok"] is True
    assert result["object"]["semantic_features"] == [definition]
    requests = result["object"]["drawing_dimension_requests"]
    assert [item["dimension_kind"] for item in requests] == [
        "diameter",
        "countersink_diameter",
    ]
    assert all(item["feature_id"] == definition["id"] for item in requests)
    assert all(item["measurement_source"] == "projected_geometry" for item in requests)
    assert all("diameter" not in item for item in requests)


def test_get_current_drawing_discovers_single_page_opened_from_file(monkeypatch):
    document = CadDocument(
        backend="freecad",
        native_handle=SimpleNamespace(),
        adapter=SimpleNamespace(),
    )

    class FakeDrawingBackend:
        def list_drawings(self, *, document):
            return [
                {
                    "drawing_id": "drawing-native_7",
                    "page_id": "native_7",
                    "page_name": "Page",
                    "label": "Existing Drawing",
                    "managed_by_rapidcad": False,
                    "view_count": 1,
                    "item_count": 0,
                    "views": [{"id": "native_8", "view_name": "top"}],
                    "items": [],
                }
            ]

    monkeypatch.setattr(
        "rapidcadpy.drawing.create_drawing_backend",
        lambda _document: FakeDrawingBackend(),
    )
    session = CadSession(execution_mode="embedded")
    session.cad_document = document

    result = session.get_current_drawing()

    assert result["ok"] is True
    assert result["drawing"]["drawing_id"] == "drawing-native_7"
    assert result["drawing"]["managed_by_rapidcad"] is False
    assert session.current_drawing_id == "drawing-native_7"


def test_get_current_drawing_requires_selection_for_multiple_pages(monkeypatch):
    document = CadDocument(
        backend="freecad",
        native_handle=SimpleNamespace(),
        adapter=SimpleNamespace(),
    )

    class FakeDrawingBackend:
        def list_drawings(self, *, document):
            return [
                {
                    "drawing_id": f"drawing-native_{index}",
                    "page_id": f"native_{index}",
                    "page_name": f"Page{index}",
                    "label": f"Page {index}",
                    "managed_by_rapidcad": False,
                    "view_count": 0,
                    "item_count": 0,
                    "views": [],
                    "items": [],
                }
                for index in (1, 2)
            ]

    monkeypatch.setattr(
        "rapidcadpy.drawing.create_drawing_backend",
        lambda _document: FakeDrawingBackend(),
    )
    session = CadSession(execution_mode="embedded")
    session.cad_document = document

    unresolved = session.get_current_drawing()
    selected = session.select_drawing("drawing-native_2")

    assert unresolved["ok"] is False
    assert "select_drawing" in unresolved["error"]
    assert selected["ok"] is True
    assert selected["drawing"]["page_name"] == "Page2"


def test_drawing_discovery_never_creates_a_document_when_none_is_open(monkeypatch):
    session = CadSession(execution_mode="gui")
    worker_calls = []

    def attach_freecad(*, use_active_document):
        session._worker = SimpleNamespace(
            call=lambda operation, params: worker_calls.append((operation, params))
        )
        return {
            "ok": True,
            "active_document": None,
            "warning": "FreeCAD has no active document.",
        }

    monkeypatch.setattr(
        session,
        "attach_freecad",
        attach_freecad,
    )
    result = session._ensure_gui_session(
        require_document=True,
        create_document_if_missing=False,
    )

    assert result["ok"] is False
    assert "Open the intended CAD file" in result["error"]
    assert worker_calls == []


def test_drawing_discovery_returns_only_runtime_registered_view_ids(monkeypatch):
    class FakeAdapter:
        backend_name = "freecad"
        parameter_adapter = None

        def recompute(self, document):
            document.recompute()

        def get_shape(self, obj):
            return None

    class NativeObject:
        def __init__(self, name, type_id):
            self.Name = name
            self.Label = name
            self.TypeId = type_id
            self.PropertiesList = []
            self.InList = []
            self.OutList = []
            self.Group = []
            self.ViewObject = SimpleNamespace(Visibility=True)

        def getParentGeoFeatureGroup(self):
            return None

    page = NativeObject("Page", "TechDraw::DrawPage")
    view = NativeObject("TopView", "TechDraw::DrawViewPart")
    native_document = SimpleNamespace(
        Name="DrawingFile",
        Label="Drawing File",
        FileName="",
        Objects=[page, view],
        recompute=lambda: None,
    )
    document = CadDocument(
        backend="freecad",
        native_handle=native_document,
        adapter=FakeAdapter(),
    )

    class FakeDrawingBackend:
        def _inspection(self):
            return {
                "page_id": document.object_id("Page"),
                "views": [
                    {
                        "id": document.object_id("TopView"),
                        "native_name": "TopView",
                        "view_name": "top",
                    }
                ],
                "items": [],
            }

        def list_drawings(self, *, document):
            inspection = self._inspection()
            return [
                {
                    "drawing_id": "drawing-page",
                    "page_id": inspection["page_id"],
                    "page_name": "Page",
                    "label": "Page",
                    "managed_by_rapidcad": False,
                    "view_count": 1,
                    "item_count": 0,
                    "views": inspection["views"],
                    "items": [],
                }
            ]

        def inspect_drawing(self, *, document, page_name):
            return self._inspection()

    monkeypatch.setattr(
        "rapidcadpy.drawing.create_drawing_backend",
        lambda _document: FakeDrawingBackend(),
    )
    session = CadSession(execution_mode="embedded")
    session.cad_document = document

    result = session.list_drawing_items()
    view_id = result["views"][0]["id"]

    assert result["ok"] is True
    assert result["view_lookup"] == {"top": view_id}
    assert f"Available views: top={view_id}" in result["summary"]
    assert view_id in session.runtime_objects
    assert session.runtime_objects[view_id].native_name == "TopView"

    resolved = session._resolve_drawing_view(
        session.drawings["drawing-page"],
        view_name="top",
        view_id=None,
    )
    assert resolved is session.runtime_objects[view_id]

    unknown = session._resolve_drawing_view(
        session.drawings["drawing-page"],
        view_name=None,
        view_id="view_1",
    )
    assert unknown["ok"] is False
    assert f"top': '{view_id}" in unknown["error"]


def test_loft_uses_ordered_registered_workplanes() -> None:
    adapter = SimpleNamespace()
    document = CadDocument(
        backend="freecad",
        native_handle=SimpleNamespace(),
        adapter=adapter,
    )
    created_shape = SimpleNamespace(
        feature=CadFeature(
            id="temporary",
            document=document,
            native_handle=SimpleNamespace(Name="Loft_1", TypeId="Part::Loft"),
            native_name="Loft_1",
            native_type="Part::Loft",
        )
    )
    first_workplane = SimpleNamespace()
    second_workplane = SimpleNamespace()
    calls = []

    def create_loft(profiles, *, make_solid, ruled):
        calls.append((profiles, make_solid, ruled))
        return created_shape

    first_workplane.loft = create_loft
    session = CadSession()
    session.backend_name = "freecad"
    session.app = SimpleNamespace()
    session.runtime_objects.update(
        {
            "workplane_1": first_workplane,
            "workplane_2": second_workplane,
        }
    )
    session.objects.update(
        {
            "workplane_1": SemanticObject(
                id="workplane_1",
                type="workplane",
                label="XY workplane",
                source_op="operation_1",
            ),
            "workplane_2": SemanticObject(
                id="workplane_2",
                type="workplane",
                label="XY workplane",
                source_op="operation_2",
            ),
        }
    )
    session._geometry_signature = lambda shape: {"volume": 42.0}  # type: ignore[method-assign]

    result = session.loft(["workplane_1", "workplane_2"], ruled=True)

    assert result["ok"] is True, result
    assert result["object_id"] == "shape_1"
    assert calls == [([second_workplane], True, True)]
    assert isinstance(session.runtime_objects["shape_1"], CadFeature)
    assert session.objects["shape_1"].metadata["profile_workplane_ids"] == [
        "workplane_1",
        "workplane_2",
    ]


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


def test_generate_drawing_result_embeds_view_lookup_without_a_second_call(
    tmp_path,
    monkeypatch,
):
    """Agents should not need list_drawing_items right after generate_drawing."""

    class FakeAdapter:
        backend_name = "freecad"
        parameter_adapter = None

        def recompute(self, document):
            document.recompute()

    native_document = SimpleNamespace(
        Name="Bracket",
        Label="Bracket",
        FileName="",
        Objects=[],
        recompute=lambda: None,
    )
    cad_document = CadDocument(
        backend="freecad",
        native_handle=native_document,
        adapter=FakeAdapter(),
        name="Bracket",
        label="Bracket",
    )

    class FakeDrawingBackend:
        def generate_drawing(self, **kwargs):
            pdf = kwargs["output_directory"] / "Bracket_run-2.pdf"
            pdf.write_bytes(b"%PDF drawing")
            return DrawingResult(pdf_path=pdf, page_name="RapidCADDrawing")

        def inspect_drawing(self, *, document, page_name):
            assert page_name == "RapidCADDrawing"
            return {
                "page_name": page_name,
                "page_id": "native_9",
                "views": [
                    {"id": "native_10", "native_name": "Front", "view_name": "front"},
                    {"id": "native_11", "native_name": "Top", "view_name": "top"},
                ],
                "items": [],
            }

    monkeypatch.setattr(
        "rapidcadpy.drawing.create_drawing_backend",
        lambda document: FakeDrawingBackend(),
    )
    session = CadSession(execution_mode="embedded")
    session.backend_name = "freecad"
    session.app = SimpleNamespace(cad_document=cad_document)
    session.cad_document = cad_document

    result = session.generate_drawing(
        output_directory=str(tmp_path),
        part_name="Bracket",
        run_id="run-2",
    )

    assert result["ok"] is True
    assert result["view_lookup"] == {"front": "native_10", "top": "native_11"}
    assert len(result["views"]) == 2
    assert "front=native_10" in result["summary"]
    assert session.drawings["run-2"]["view_lookup"] == result["view_lookup"]


def test_resolve_drawing_view_recovers_a_guessed_semantic_id_from_view_id():
    """A view_id like 'FrontView' should resolve via the view_name lookup."""

    session = CadSession()
    document = CadDocument(
        backend="freecad",
        native_handle=SimpleNamespace(),
        adapter=SimpleNamespace(),
    )
    view_object = CadFeature(
        id="native_10",
        document=document,
        native_handle=SimpleNamespace(Name="Front"),
        native_name="Front",
        native_type="TechDraw::DrawViewPart",
    )
    session.runtime_objects["native_10"] = view_object
    drawing = {"views": [{"id": "native_10", "view_name": "front"}]}

    resolved = session._resolve_drawing_view(
        drawing, view_name=None, view_id="FrontView"
    )

    assert resolved is view_object

    still_unknown = session._resolve_drawing_view(
        drawing, view_name=None, view_id="SideView"
    )
    assert still_unknown["ok"] is False
    assert "Unknown drawing view_id" in still_unknown["error"]
