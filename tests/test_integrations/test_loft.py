import pytest


class TestLoft:
    """Reusable loft tests for backend-specific integration suites."""

    @pytest.fixture
    def app(self):
        app = "freecad"
        if app == "freecad":
            from vendor.rapidcadpy.rapidcadpy.integrations.freecad import FreeCADApp

            app = FreeCADApp()
            yield app
        elif app == "ocp":
            from vendor.rapidcadpy.rapidcadpy.integrations.freecad import (
                OpenCascadeOCPApp,
            )

            app = OpenCascadeOCPApp()
            yield app

    def test_loft_simple(self, app):
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
        if "freecad" in app.__class__.lower():
            shape.to_fcstd("outputs/minimal_loft.fcstd")
        else:
            shape.to_step("outputs/minimal_loft.step")

    def test_loft_circular_profiles(self, app):
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
        shape.to_fcstd("outputs/circular_loft.fcstd")
