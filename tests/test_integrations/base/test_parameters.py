"""
Base for named-parameter contract tests.

Exports
-------
_dummy_workplane     Helper that creates a minimal Workplane bound to an app.
_BaseParameterTests  Mixin with all backend-agnostic test methods.
TestBaseAppParameters  Concrete class for the raw dict-backed ``App`` base class.

Each integration folder imports ``_BaseParameterTests`` and provides its own
``app`` fixture — that is the only thing that varies between backends.
"""

import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _dummy_workplane(app):
    """Return a minimal Workplane subclass instance bound to *app*.

    Sufficient for testing ``_resolve_distance`` because that method lives on
    the base ``Workplane`` and only calls ``self.app.get_parameter()``.
    """
    from rapidcadpy.workplane import Workplane

    class _Dummy(Workplane):
        @classmethod
        def create_offset_plane(cls, *a, **k):
            pass

    wp = _Dummy.__new__(_Dummy)
    wp.app = app
    wp._pending_shapes = []
    return wp


# ---------------------------------------------------------------------------
# Mixin — all backend-agnostic tests live here
# ---------------------------------------------------------------------------


class _BaseParameterTests:
    """Subclasses must provide an ``app`` pytest fixture that yields a
    ready-to-use App instance (document already open where applicable)."""

    # ------------------------------------------------------------------
    # add_parameter
    # ------------------------------------------------------------------

    def test_add_returns_float(self, app):
        val = app.add_parameter("width", 50)
        assert isinstance(val, float)

    def test_add_stores_value(self, app):
        app.add_parameter("height", 30)
        assert app.get_parameter("height") == pytest.approx(30.0, rel=0.01)

    def test_add_appears_in_list(self, app):
        app.add_parameter("depth", 10)
        assert "depth" in app.list_parameters()

    def test_add_overwrites_existing(self, app):
        app.add_parameter("width", 50)
        app.add_parameter("width", 80)
        assert app.get_parameter("width") == pytest.approx(80.0, rel=0.01)
        assert list(app.list_parameters().keys()).count("width") == 1

    # ------------------------------------------------------------------
    # get_parameter
    # ------------------------------------------------------------------

    def test_get_returns_added_value(self, app):
        added = app.add_parameter("length", 75.0)
        assert app.get_parameter("length") == pytest.approx(added, rel=0.01)

    def test_get_missing_raises_key_error(self, app):
        with pytest.raises(KeyError):
            app.get_parameter("__no_such_param__")

    # ------------------------------------------------------------------
    # set_parameter
    # ------------------------------------------------------------------

    def test_set_updates_value(self, app):
        app.add_parameter("gap", 20)
        app.set_parameter("gap", 40)
        assert app.get_parameter("gap") == pytest.approx(40.0, rel=0.01)

    def test_set_missing_raises_key_error(self, app):
        with pytest.raises(KeyError):
            app.set_parameter("__no_such_param__", 10)

    # ------------------------------------------------------------------
    # list_parameters
    # ------------------------------------------------------------------

    def test_list_empty_on_fresh_instance(self, app):
        assert app.list_parameters() == {}

    def test_list_returns_all_added(self, app):
        app.add_parameter("a", 1)
        app.add_parameter("b", 2)
        app.add_parameter("c", 3)
        assert {"a", "b", "c"} <= app.list_parameters().keys()

    # ------------------------------------------------------------------
    # Round-trip
    # ------------------------------------------------------------------

    def test_round_trip_add_get(self, app):
        added = app.add_parameter("wall", 5)
        assert app.get_parameter("wall") == pytest.approx(added, rel=0.01)

    def test_round_trip_add_set_get(self, app):
        app.add_parameter("fillet_r", 3)
        app.set_parameter("fillet_r", 6)
        assert app.get_parameter("fillet_r") == pytest.approx(6.0, rel=0.01)

    # ------------------------------------------------------------------
    # _resolve_distance  (defined on base Workplane, same for all backends)
    # ------------------------------------------------------------------

    def test_resolve_float_passthrough(self, app):
        assert _dummy_workplane(app)._resolve_distance(15.0) == pytest.approx(15.0)

    def test_resolve_int_passthrough(self, app):
        assert _dummy_workplane(app)._resolve_distance(7) == pytest.approx(7.0)

    def test_resolve_string_looks_up_parameter(self, app):
        app.add_parameter("extrude_depth", 25.0)
        expected = app.get_parameter("extrude_depth")
        assert _dummy_workplane(app)._resolve_distance(
            "extrude_depth"
        ) == pytest.approx(expected, rel=0.01)

    def test_resolve_unknown_string_raises_key_error(self, app):
        with pytest.raises(KeyError):
            _dummy_workplane(app)._resolve_distance("__no_such_param__")

    def test_resolve_invalid_type_raises_type_error(self, app):
        with pytest.raises(TypeError):
            _dummy_workplane(app)._resolve_distance([10, 20])


# ---------------------------------------------------------------------------
# Concrete: plain dict-backed base App  (no geometry library required)
# ---------------------------------------------------------------------------


class TestBaseAppParameters(_BaseParameterTests):
    """Parameter tests against the raw ``App`` base class."""

    @pytest.fixture
    def app(self):
        from rapidcadpy.app import App

        return App()
