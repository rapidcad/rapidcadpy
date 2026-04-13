"""
Integration tests for InventorApp named parameter API.

All tests run against a live, connected Inventor instance.  The entire module
is skipped automatically when win32com / Inventor is not available (macOS,
Linux, CI without Inventor).

Tests cover:
  - add_parameter()   – create a new user parameter
  - get_parameter()   – read by name
  - set_parameter()   – update an existing parameter
  - list_parameters() – enumerate all user parameters
  - expression-based parameters (driven by a formula string)
  - passing a parameter name string directly to _resolve_distance()
  - error cases: unknown unit, missing document, missing parameter name
"""

import pytest

# Skip the whole module if win32com / Inventor is not installed.
pytest.importorskip("win32com.client", reason="win32com / Inventor not available")


# ---------------------------------------------------------------------------
# Fixture: real InventorApp with a fresh document per test
# ---------------------------------------------------------------------------


@pytest.fixture()
def app():
    from rapidcadpy.integrations.inventor.app import InventorApp

    instance = InventorApp()
    instance.new_document()
    yield instance
    try:
        instance.close_document()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests: add_parameter
# ---------------------------------------------------------------------------


class TestAddParameter:
    def test_add_simple_mm_parameter(self, app):
        val = app.add_parameter("width", 50)
        # Return value should match what Inventor stored
        assert val == pytest.approx(app.get_parameter("width"), rel=0.01)

    def test_add_parameter_appears_in_user_params(self, app):
        app.add_parameter("height", 30)
        assert "height" in app.list_parameters()

    def test_add_parameter_with_inch_units(self, app):
        val = app.add_parameter("radius", 2.5, units="in")
        assert val == pytest.approx(app.get_parameter("radius"), rel=0.01)

    def test_add_parameter_with_expression(self, app):
        app.add_parameter("base_width", 100)
        app.add_parameter("flange", 0, expression="base_width / 4")
        # Inventor evaluates the expression: flange = base_width / 4
        base_val = app.get_parameter("base_width")
        flange_val = app.get_parameter("flange")
        assert flange_val == pytest.approx(base_val / 4, rel=0.01)

    def test_add_parameter_updates_existing(self, app):
        app.add_parameter("width", 50)
        val_50 = app.get_parameter("width")
        app.add_parameter("width", 80)  # should update, not create a second entry
        val_80 = app.get_parameter("width")
        # Value must have changed
        assert val_80 != pytest.approx(val_50, rel=0.01)
        # Only one "width" parameter should exist
        assert list(app.list_parameters().keys()).count("width") == 1

    def test_add_parameter_unknown_unit_raises(self, app):
        with pytest.raises(ValueError, match="Unknown unit"):
            app.add_parameter("x", 10, units="parsec")

    def test_add_parameter_requires_open_document(self, app):
        app.comp_def = None
        with pytest.raises(RuntimeError, match="No active document"):
            app.add_parameter("x", 5)

    def test_add_parameter_and_save_file(self, app, tmp_path):
        # Verify that parameters persist in the file and can be read back after reopening
        app.add_parameter("thickness", 12.5)
        original_val = app.get_parameter("thickness")  # internal units (cm)
        file_path = tmp_path / "test_params.ipt"
        app.to_ipt(str(file_path))
        print(f"Saved file to: {file_path}")

        # Reopen the file in a new InventorApp instance
        from rapidcadpy.integrations.inventor.app import InventorApp

        new_app = InventorApp()
        new_app.open_document(str(file_path))
        assert new_app.get_parameter("thickness") == pytest.approx(original_val, rel=0.01)


# ---------------------------------------------------------------------------
# Tests: get_parameter
# ---------------------------------------------------------------------------


class TestGetParameter:
    def test_get_existing_parameter(self, app):
        added = app.add_parameter("length", 75.0)
        assert app.get_parameter("length") == pytest.approx(added, rel=0.01)

    def test_get_parameter_not_found_raises(self, app):
        with pytest.raises(KeyError, match="nonexistent"):
            app.get_parameter("nonexistent")

    def test_get_parameter_searches_model_params(self, app):
        # Model parameters are driven dimensions attached to geometry; we
        # cannot create them without a sketch.  Verify that the search falls
        # through both UserParameters and ModelParameters before raising.
        with pytest.raises(KeyError):
            app.get_parameter("d0_no_such_model_param")


# ---------------------------------------------------------------------------
# Tests: set_parameter
# ---------------------------------------------------------------------------


class TestSetParameter:
    def test_set_existing_parameter_updates_value(self, app):
        app.add_parameter("depth", 20)
        val_20 = app.get_parameter("depth")
        app.set_parameter("depth", 40)
        val_40 = app.get_parameter("depth")
        assert val_40 != pytest.approx(val_20, rel=0.01)

    def test_set_parameter_with_different_units(self, app):
        # 10 mm == 1 cm — value in Inventor internal units should stay the same
        app.add_parameter("gap", 10, units="mm")
        val_mm = app.get_parameter("gap")
        app.set_parameter("gap", 1, units="cm")
        val_cm = app.get_parameter("gap")
        assert val_cm == pytest.approx(val_mm, rel=0.01)

    def test_set_parameter_not_found_raises(self, app):
        with pytest.raises(KeyError, match="missing_param"):
            app.set_parameter("missing_param", 10)


# ---------------------------------------------------------------------------
# Tests: list_parameters
# ---------------------------------------------------------------------------


class TestListParameters:
    def test_list_empty(self, app):
        # A brand-new document has no user parameters
        assert app.list_parameters() == {}

    def test_list_multiple_parameters(self, app):
        app.add_parameter("width", 50.0)
        app.add_parameter("height", 30.0)
        app.add_parameter("depth", 10.0)
        result = app.list_parameters()
        assert "width" in result
        assert "height" in result
        assert "depth" in result

    def test_list_parameters_no_document(self, app):
        app.comp_def = None
        assert app.list_parameters() == {}


# ---------------------------------------------------------------------------
# Tests: round-trip  (add → get → set → get)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_add_then_get(self, app):
        added = app.add_parameter("wall_thickness", 5)
        assert app.get_parameter("wall_thickness") == pytest.approx(added, rel=0.01)

    def test_add_set_get(self, app):
        app.add_parameter("fillet_r", 3)
        val_before = app.get_parameter("fillet_r")
        app.set_parameter("fillet_r", 6)
        val_after = app.get_parameter("fillet_r")
        assert val_after != pytest.approx(val_before, rel=0.01)

    def test_driven_expression_add_then_list(self, app):
        app.add_parameter("outer_r", 20)
        app.add_parameter("inner_r", 0, expression="outer_r / 2")
        params = app.list_parameters()
        assert "outer_r" in params
        assert "inner_r" in params
        assert params["inner_r"] == pytest.approx(params["outer_r"] / 2, rel=0.01)


# ---------------------------------------------------------------------------
# Tests: _resolve_distance() accepting parameter name strings
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

        app.add_parameter("extrude_depth", 25.0)
        expected = app.get_parameter("extrude_depth")

        wp = object.__new__(InventorWorkPlane)
        wp.app = app
        assert wp._resolve_distance("extrude_depth") == pytest.approx(expected, rel=0.01)

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
