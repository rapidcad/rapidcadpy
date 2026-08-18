from types import SimpleNamespace

import pytest

from rapidcadpy.integrations.freecad.sketch2d import (
    FreeCADNativeFeatureError,
    FreeCADSketch2D,
)


class _FakeDocument:
    def __init__(self) -> None:
        self.added_types = []
        self.removed_names = []

    def addObject(self, type_id, name):
        self.added_types.append(type_id)
        return SimpleNamespace(Name=name)

    def removeObject(self, name):
        self.removed_names.append(name)

    def recompute(self):
        return None


class _FakeApp:
    def __init__(self, document) -> None:
        self.document = document

    def get_doc(self):
        return self.document

    def get_next_feature_index(self):
        return 1


def test_native_extrusion_failure_does_not_fall_back_to_part_feature(monkeypatch):
    document = _FakeDocument()
    sketch = FreeCADSketch2D(
        primitives=[object()],
        workplane=object(),
        app=_FakeApp(document),
    )
    monkeypatch.setattr(
        sketch,
        "_create_editable_sketch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("planned failure")),
    )

    with pytest.raises(FreeCADNativeFeatureError, match="refusing"):
        sketch._apply_operation(object(), "NewBodyFeatureOperation")

    assert "Part::Feature" not in document.added_types


@pytest.mark.parametrize(
    "operation",
    ["JoinBodyFeatureOperation", "Cut", "CutOperation", "Intersect"],
)
def test_unsupported_extrusion_operations_fail_instead_of_baking(operation):
    sketch = FreeCADSketch2D(
        primitives=[object()],
        workplane=object(),
        app=None,
    )

    with pytest.raises(FreeCADNativeFeatureError, match="not implemented"):
        sketch._apply_operation(object(), operation)
