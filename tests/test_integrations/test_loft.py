import pytest

pytest.importorskip("FreeCAD", reason="requires the FreeCAD Python runtime")

from rapidcadpy.integrations.freecad import FreeCADApp


class TestLoft:
    """Reusable loft tests for backend-specific integration suites."""

    @pytest.fixture
    def app(self):
        yield FreeCADApp()

    def test_loft_simple(self, app, tmp_path):
        wp1 = app.work_plane("XY", offset=0)
        wp1.rect(50, 30, centered=False).close()

        wp2 = app.work_plane("XY", offset=40)
        wp2.move_to(25, 15).circle(15).close()

        shape = wp1.loft(wp2, make_solid=True, ruled=False)
        doc = app.get_doc()
        loft_objs = [obj for obj in doc.Objects if obj.TypeId == "Part::Loft"]
        assert loft_objs, "Expected a Part::Loft feature in FreeCAD document"
        sketch_objs = [
            obj for obj in doc.Objects if obj.TypeId == "Sketcher::SketchObject"
        ]
        assert sketch_objs, "Expected editable Sketcher sections in FreeCAD document"
        shape.to_fcstd(str(tmp_path / "minimal_loft.fcstd"))

    def test_loft_circular_profiles(self, app, tmp_path):
        wp1 = app.work_plane("XY", offset=0)
        wp1.circle(15).close()

        wp2 = app.work_plane("XY", offset=40)
        wp2.circle(25).close()

        shape = wp1.loft(wp2, make_solid=True, ruled=False)
        doc = app.get_doc()
        sketch_objs = [
            obj for obj in doc.Objects if obj.TypeId == "Sketcher::SketchObject"
        ]
        assert sketch_objs, "Expected editable Sketcher sections in FreeCAD document"
        shape.to_fcstd(str(tmp_path / "circular_loft.fcstd"))
