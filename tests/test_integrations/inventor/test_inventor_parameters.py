"""Named-parameter tests for the Autodesk Inventor backend.

Skipped automatically when win32com / Inventor is not available (macOS / CI).
Inherits the full backend-agnostic contract from _BaseParameterTests and adds
Inventor-specific tests: expression evaluation, unit conversion, document
lifecycle guards, native InventorWorkPlane resolve, and file round-trip.
"""

import pytest

from ..base.test_parameters import _BaseParameterTests

_win32com_available: bool
try:
    import win32com.client  # noqa: F401

    _win32com_available = True
except ImportError:
    _win32com_available = False


@pytest.mark.skipif(not _win32com_available, reason="win32com / Inventor not available")
class TestInventorParameters(_BaseParameterTests):
    """Full parameter contract + Inventor-specific extras."""

    @pytest.fixture
    def app(self):
        from rapidcadpy.integrations.inventor.app import InventorApp

        instance = InventorApp()
        instance.new_document()
        yield instance
        try:
            instance.close_document()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Expression-driven parameters (Inventor evaluates the formula)
    # ------------------------------------------------------------------

    def test_expression_evaluates_correctly(self, app):
        app.add_parameter("base_width", 100)
        app.add_parameter("flange", 0, expression="base_width / 4")
        base_val = app.get_parameter("base_width")
        assert app.get_parameter("flange") == pytest.approx(base_val / 4, rel=0.01)

    def test_expression_round_trip_in_list(self, app):
        app.add_parameter("outer_r", 20)
        app.add_parameter("inner_r", 0, expression="outer_r / 2")
        params = app.list_parameters()
        assert params["inner_r"] == pytest.approx(params["outer_r"] / 2, rel=0.01)

    # ------------------------------------------------------------------
    # Unit validation (Inventor rejects unknown unit strings)
    # ------------------------------------------------------------------

    def test_unknown_unit_raises(self, app):
        with pytest.raises(ValueError, match="Unknown unit"):
            app.add_parameter("x", 10, units="parsec")

    def test_set_mm_then_cm_same_internal_value(self, app):
        # 10 mm == 1 cm; Inventor stores in internal units so both should match
        app.add_parameter("gap", 10, units="mm")
        val_mm = app.get_parameter("gap")
        app.set_parameter("gap", 1, units="cm")
        assert app.get_parameter("gap") == pytest.approx(val_mm, rel=0.01)

    # ------------------------------------------------------------------
    # Document lifecycle guards
    # ------------------------------------------------------------------

    def test_add_requires_open_document(self, app):
        app.comp_def = None
        with pytest.raises(RuntimeError, match="No active document"):
            app.add_parameter("x", 5)

    def test_list_returns_empty_without_document(self, app):
        app.comp_def = None
        assert app.list_parameters() == {}

    def test_get_searches_model_parameters_before_raising(self, app):
        with pytest.raises(KeyError):
            app.get_parameter("d0_no_such_model_param")

    # ------------------------------------------------------------------
    # _resolve_distance via native InventorWorkPlane
    # ------------------------------------------------------------------

    def test_inventor_workplane_resolve_string(self, app):
        from rapidcadpy.integrations.inventor.workplane import InventorWorkPlane

        app.add_parameter("extrude_depth", 25.0)
        expected = app.get_parameter("extrude_depth")
        wp = object.__new__(InventorWorkPlane)
        wp.app = app
        assert wp._resolve_distance("extrude_depth") == pytest.approx(
            expected, rel=0.01
        )

    # ------------------------------------------------------------------
    # File save / reload round-trip
    # ------------------------------------------------------------------

    def test_parameter_persists_after_save_and_reload(self, app, tmp_path):
        app.add_parameter("thickness", 12.5)
        original_val = app.get_parameter("thickness")
        file_path = tmp_path / "test_params.ipt"
        app.to_ipt(str(file_path))

        from rapidcadpy.integrations.inventor.app import InventorApp

        new_app = InventorApp()
        new_app.open_document(str(file_path))
        assert new_app.get_parameter("thickness") == pytest.approx(
            original_val, rel=0.01
        )
