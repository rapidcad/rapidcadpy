"""
Unit tests for InventorApp named parameter API.

Tests cover:
  - add_parameter()  – create a new user parameter
  - get_parameter()  – read by name
  - set_parameter()  – update an existing parameter
  - list_parameters() – enumerate all user parameters
  - expression-based parameters (driven by a formula string)
  - passing a parameter name string directly to extrude() / revolve()
  - error cases: unknown unit, missing document, missing parameter name

The COM layer (win32com / pywin32) is fully mocked so these tests run on any
platform (macOS, Linux, CI) without a real Inventor installation.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build the minimal fake COM hierarchy
# ---------------------------------------------------------------------------


def _make_param(name: str, value: float, expression: str = "") -> MagicMock:
    """Return a mock Inventor Parameter object."""
    p = MagicMock()
    p.Name = name
    p.Value = value
    p.Expression = expression or f"{value} mm"
    return p


def _make_user_params(params: list) -> MagicMock:
    """Return a mock UserParameters collection from a list of mock params."""
    coll = MagicMock()
    coll.Count = len(params)
    # Item() is 1-based
    coll.Item = MagicMock(side_effect=lambda i: params[i - 1])

    def _add_by_expression(name, expr, units_enum):
        p = _make_param(name, 0.0, expr)
        # simulate Inventor evaluating the expression to a numeric value:
        # parse simple "<number> <unit>" or pure number
        try:
            p.Value = float(expr.split()[0])
        except Exception:
            p.Value = 0.0
        params.append(p)
        coll.Count = len(params)
        return p

    coll.AddByExpression = MagicMock(side_effect=_add_by_expression)
    return coll


def _make_comp_def(user_params_list: list | None = None) -> MagicMock:
    """Return a mock ComponentDefinition with Parameters."""
    if user_params_list is None:
        user_params_list = []
    user_params = _make_user_params(user_params_list)

    model_params = MagicMock()
    model_params.Count = 0

    params_container = MagicMock()
    params_container.UserParameters = user_params
    params_container.ModelParameters = model_params

    comp_def = MagicMock()
    comp_def.Parameters = params_container
    return comp_def


# ---------------------------------------------------------------------------
# Fixture: a bare InventorApp with mocked COM (no real Inventor needed)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_inventor_modules():
    """
    Inject fake win32com / pythoncom / win32com.client into sys.modules
    so InventorApp can be imported and instantiated on non-Windows hosts.
    """
    # Build a minimal fake win32com.client module
    win32_client = MagicMock()
    win32_client.GetActiveObject = MagicMock(side_effect=Exception("no COM"))
    win32_client.Dispatch = MagicMock(return_value=MagicMock())
    win32_client.CastTo = MagicMock(side_effect=lambda obj, _cls: obj)
    win32_client.constants = MagicMock()
    win32_client.gencache = MagicMock()
    win32_client.gencache.EnsureModule = MagicMock(return_value=None)

    win32_mod = ModuleType("win32com")
    win32_mod.client = win32_client

    pythoncom = MagicMock()

    old = {}
    for key in ("win32com", "win32com.client", "pythoncom"):
        old[key] = sys.modules.get(key)

    sys.modules["win32com"] = win32_mod
    sys.modules["win32com.client"] = win32_client
    sys.modules["pythoncom"] = pythoncom

    yield win32_client

    # Restore
    for key, val in old.items():
        if val is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = val


@pytest.fixture()
def app(mock_inventor_modules):
    """
    Return an InventorApp whose __init__ is bypassed and whose comp_def is
    pre-populated with a fresh empty user-parameter store.
    """
    # Import after the fake modules are in place
    from rapidcadpy.integrations.inventor.app import InventorApp, _UNIT_MAP

    _params_list: list = []
    comp_def = _make_comp_def(_params_list)

    instance = object.__new__(InventorApp)
    # Minimal attributes set by App.__init__ / InventorApp.__init__
    instance.comp_def = comp_def
    instance._params_list = _params_list  # keep reference for assertions
    instance._shapes = []

    # Provide a minimal transient_obj for thread / chamfer tests (not exercised here)
    instance.transient_obj = MagicMock()
    instance.inventor_app = MagicMock()

    return instance


# ---------------------------------------------------------------------------
# Tests: add_parameter
# ---------------------------------------------------------------------------


class TestAddParameter:
    def test_add_simple_mm_parameter(self, app):
        val = app.add_parameter("width", 50)
        assert val == pytest.approx(50.0)

    def test_add_parameter_appears_in_user_params(self, app):
        app.add_parameter("height", 30)
        # AddByExpression should have been called once
        app.comp_def.Parameters.UserParameters.AddByExpression.assert_called_once()
        call_args = app.comp_def.Parameters.UserParameters.AddByExpression.call_args
        assert call_args[0][0] == "height"
        assert "30" in call_args[0][1]  # expression contains the value

    def test_add_parameter_with_inch_units(self, app):
        val = app.add_parameter("radius", 2.5, units="in")
        assert val == pytest.approx(2.5)

    def test_add_parameter_with_expression(self, app):
        # First add a base parameter so the expression is meaningful
        app.add_parameter("base_width", 100)
        val = app.add_parameter("flange", 0, expression="base_width / 4")
        # AddByExpression should have been called twice
        assert app.comp_def.Parameters.UserParameters.AddByExpression.call_count == 2
        call_args = app.comp_def.Parameters.UserParameters.AddByExpression.call_args
        assert call_args[0][1] == "base_width / 4"

    def test_add_parameter_updates_existing(self, app):
        """If a param with the same name already exists, its Expression is updated."""
        # Pre-populate the params list with an existing param
        existing = _make_param("width", 50.0, "50 mm")
        app._params_list.append(existing)
        app.comp_def.Parameters.UserParameters.Count = 1
        app.comp_def.Parameters.UserParameters.Item = MagicMock(
            side_effect=lambda i: app._params_list[i - 1]
        )

        app.add_parameter("width", 80)

        # Should NOT call AddByExpression again
        app.comp_def.Parameters.UserParameters.AddByExpression.assert_not_called()
        # Should update the existing parameter's Expression
        assert "80" in existing.Expression

    def test_add_parameter_unknown_unit_raises(self, app):
        with pytest.raises(ValueError, match="Unknown unit"):
            app.add_parameter("x", 10, units="parsec")

    def test_add_parameter_requires_open_document(self, app):
        app.comp_def = None
        with pytest.raises(RuntimeError, match="No active document"):
            app.add_parameter("x", 5)


# ---------------------------------------------------------------------------
# Tests: get_parameter
# ---------------------------------------------------------------------------


class TestGetParameter:
    def test_get_existing_parameter(self, app):
        app._params_list.append(_make_param("length", 75.0, "75 mm"))
        app.comp_def.Parameters.UserParameters.Count = 1
        app.comp_def.Parameters.UserParameters.Item = MagicMock(
            side_effect=lambda i: app._params_list[i - 1]
        )

        val = app.get_parameter("length")
        assert val == pytest.approx(75.0)

    def test_get_parameter_not_found_raises(self, app):
        with pytest.raises(KeyError, match="nonexistent"):
            app.get_parameter("nonexistent")

    def test_get_parameter_from_model_params_fallback(self, app):
        """Parameters not in UserParameters should be found in ModelParameters."""
        model_param = _make_param("d0", 12.0, "12 mm")
        model_params = MagicMock()
        model_params.Count = 1
        model_params.Item = MagicMock(return_value=model_param)
        app.comp_def.Parameters.ModelParameters = model_params

        val = app.get_parameter("d0")
        assert val == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Tests: set_parameter
# ---------------------------------------------------------------------------


class TestSetParameter:
    def test_set_existing_parameter_updates_expression(self, app):
        p = _make_param("depth", 20.0, "20 mm")
        app._params_list.append(p)
        app.comp_def.Parameters.UserParameters.Count = 1
        app.comp_def.Parameters.UserParameters.Item = MagicMock(
            side_effect=lambda i: app._params_list[i - 1]
        )

        app.set_parameter("depth", 40)

        assert "40" in p.Expression  # value embedded in expression

    def test_set_parameter_with_different_units(self, app):
        p = _make_param("gap", 5.0, "5 mm")
        app._params_list.append(p)
        app.comp_def.Parameters.UserParameters.Count = 1
        app.comp_def.Parameters.UserParameters.Item = MagicMock(
            side_effect=lambda i: app._params_list[i - 1]
        )

        app.set_parameter("gap", 0.5, units="cm")
        assert "0.5" in p.Expression
        assert "cm" in p.Expression

    def test_set_parameter_not_found_raises(self, app):
        with pytest.raises(KeyError, match="missing_param"):
            app.set_parameter("missing_param", 10)


# ---------------------------------------------------------------------------
# Tests: list_parameters
# ---------------------------------------------------------------------------


class TestListParameters:
    def test_list_empty(self, app):
        result = app.list_parameters()
        assert result == {}

    def test_list_multiple_parameters(self, app):
        params = [
            _make_param("width", 50.0),
            _make_param("height", 30.0),
            _make_param("depth", 10.0),
        ]
        app._params_list.extend(params)
        app.comp_def.Parameters.UserParameters.Count = 3
        app.comp_def.Parameters.UserParameters.Item = MagicMock(
            side_effect=lambda i: app._params_list[i - 1]
        )

        result = app.list_parameters()

        assert result == {"width": 50.0, "height": 30.0, "depth": 10.0}

    def test_list_parameters_no_document(self, app):
        app.comp_def = None
        result = app.list_parameters()
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: round-trip  (add → get → set → get)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_add_then_get(self, app):
        added = app.add_parameter("wall_thickness", 5)
        fetched = app.get_parameter("wall_thickness")
        assert added == pytest.approx(fetched)

    def test_add_set_get(self, app):
        app.add_parameter("fillet_r", 3)
        # After add the param is in _params_list
        # Rewire Item so get/set can find it
        app.comp_def.Parameters.UserParameters.Item = MagicMock(
            side_effect=lambda i: app._params_list[i - 1]
        )

        app.set_parameter("fillet_r", 6)
        val = app.get_parameter("fillet_r")
        # The mock updates Expression, not Value, so we just verify no exception
        # and that the expression was updated
        assert "6" in app._params_list[-1].Expression

    def test_driven_expression_add_then_list(self, app):
        app.add_parameter("outer_r", 20)
        app.add_parameter("inner_r", 0, expression="outer_r / 2")
        app.comp_def.Parameters.UserParameters.Item = MagicMock(
            side_effect=lambda i: app._params_list[i - 1]
        )
        params = app.list_parameters()
        assert "outer_r" in params
        assert "inner_r" in params


# ---------------------------------------------------------------------------
# Tests: extrude() / revolve() accepting parameter name strings
# ---------------------------------------------------------------------------


class TestParameterStringInWorkplane:
    """Verify _resolve_distance() on InventorWorkPlane handles str → float."""

    def test_resolve_float_passthrough(self, app):
        from rapidcadpy.integrations.inventor.workplane import InventorWorkPlane

        wp = object.__new__(InventorWorkPlane)
        wp.app = app

        assert wp._resolve_distance(15.0) == pytest.approx(15.0)

    def test_resolve_int_passthrough(self, app):
        from rapidcadpy.integrations.inventor.workplane import InventorWorkPlane

        wp = object.__new__(InventorWorkPlane)
        wp.app = app

        assert wp._resolve_distance(7) == pytest.approx(7.0)

    def test_resolve_string_looks_up_parameter(self, app):
        from rapidcadpy.integrations.inventor.workplane import InventorWorkPlane

        # Pre-populate the named parameter
        app._params_list.append(_make_param("extrude_depth", 25.0))
        app.comp_def.Parameters.UserParameters.Count = 1
        app.comp_def.Parameters.UserParameters.Item = MagicMock(
            side_effect=lambda i: app._params_list[i - 1]
        )

        wp = object.__new__(InventorWorkPlane)
        wp.app = app

        assert wp._resolve_distance("extrude_depth") == pytest.approx(25.0)

    def test_resolve_unknown_string_raises_key_error(self, app):
        from rapidcadpy.integrations.inventor.workplane import InventorWorkPlane

        wp = object.__new__(InventorWorkPlane)
        wp.app = app

        with pytest.raises(KeyError):
            wp._resolve_distance("no_such_param")

    def test_resolve_invalid_type_raises_type_error(self, app):
        from rapidcadpy.integrations.inventor.workplane import InventorWorkPlane

        wp = object.__new__(InventorWorkPlane)
        wp.app = app

        with pytest.raises(TypeError):
            wp._resolve_distance([10, 20])


# ---------------------------------------------------------------------------
# Live Inventor tests (skipped unless win32com is genuinely importable)
# ---------------------------------------------------------------------------

_win32com_available = False
try:
    import win32com.client  # noqa: F401

    _win32com_available = True
except ImportError:
    pass


@pytest.mark.skipif(not _win32com_available, reason="win32com / Inventor not available")
class TestLiveInventorParameters:
    """
    Integration tests that require a running Inventor application.
    These are skipped on macOS / Linux and in CI environments without Inventor.
    """

    @pytest.fixture()
    def live_app(self):
        from rapidcadpy.integrations.inventor.app import InventorApp

        app = InventorApp()
        app.new_document()
        yield app
        try:
            app.close_document()
        except Exception:
            pass

    def test_live_add_and_get_parameter(self, live_app):
        val = live_app.add_parameter("live_width", 60)
        assert val == pytest.approx(60.0, abs=0.01)

        fetched = live_app.get_parameter("live_width")
        assert fetched == pytest.approx(60.0, abs=0.01)

    def test_live_set_parameter(self, live_app):
        live_app.add_parameter("live_height", 40)
        live_app.set_parameter("live_height", 80)

        val = live_app.get_parameter("live_height")
        assert val == pytest.approx(80.0, abs=0.01)

    def test_live_list_parameters(self, live_app):
        live_app.add_parameter("p_a", 10)
        live_app.add_parameter("p_b", 20)

        params = live_app.list_parameters()
        assert "p_a" in params
        assert "p_b" in params
        assert params["p_a"] == pytest.approx(10.0, abs=0.01)
        assert params["p_b"] == pytest.approx(20.0, abs=0.01)

    def test_live_driven_expression(self, live_app):
        live_app.add_parameter("outer", 50)
        live_app.add_parameter("inner", 0, expression="outer / 2")

        inner_val = live_app.get_parameter("inner")
        assert inner_val == pytest.approx(25.0, abs=0.1)

    def test_live_extrude_with_named_parameter(self, live_app):
        depth = live_app.add_parameter("box_depth", 15)
        wp = live_app.work_plane("XY")
        shape = wp.rect(20, 20).extrude("box_depth")
        assert shape is not None

    def test_live_update_parameter_rebuilds_model(self, live_app):
        """Verify that changing a parameter and rebuilding gives the updated value."""
        live_app.add_parameter("dyn_depth", 10)
        wp = live_app.work_plane("XY")
        wp.rect(10, 10).extrude("dyn_depth")

        live_app.set_parameter("dyn_depth", 20)
        live_app.inventor_app.ActiveDocument.Update()  # trigger rebuild

        # After rebuild the model should reflect the new value; we can only verify
        # the parameter value itself here without geometry queries.
        val = live_app.get_parameter("dyn_depth")
        assert val == pytest.approx(20.0, abs=0.1)
