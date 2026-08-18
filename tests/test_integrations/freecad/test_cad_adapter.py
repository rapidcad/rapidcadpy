from types import SimpleNamespace

from rapidcadpy.integrations.freecad.cad_adapter import FreeCADAdapter


def test_boolean_result_exclusively_owns_visibility() -> None:
    base = SimpleNamespace(ViewObject=SimpleNamespace(Visibility=True))
    tool = SimpleNamespace(ViewObject=SimpleNamespace(Visibility=True))
    result = SimpleNamespace(ViewObject=SimpleNamespace(Visibility=False))

    FreeCADAdapter.set_boolean_result_visibility(result, [base, tool])

    assert base.ViewObject.Visibility is False
    assert tool.ViewObject.Visibility is False
    assert result.ViewObject.Visibility is True
