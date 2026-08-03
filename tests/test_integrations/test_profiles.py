import importlib
from pathlib import Path

import pytest


def _ipe_area(section) -> float:
    return (
        2.0 * section.flange_width * section.flange_thickness
        + (section.depth - 2.0 * section.flange_thickness) * section.web_thickness
    )


class TestIpeBeam:
    @pytest.fixture
    def result_dir(self):
        path = Path("outputs/profiles")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @pytest.fixture
    def app(self):
        try:
            from ...rapidcadpy.integrations.freecad import FreeCADApp

            app = FreeCADApp()
        except ImportError as exc:
            pytest.skip(f"FreeCAD is not importable in this Python runtime: {exc}")
        yield app

    def test_profiles_public_api_exposes_ipe(self):
        from ...rapidcadpy.components import profiles

        section = profiles.ipe("IPE80")

        assert section.name == "IPE80"
        assert profiles.list_profiles()["ipe"] == ["IPE100", "IPE120", "IPE80"]

    def test_ipe_beam_extrusion_geometry(self, app, result_dir):
        from ...rapidcadpy.components import profiles

        section = profiles.ipe("IPE80")
        length = 300.0
        wp = app.work_plane("XY")

        beam = section.sketch(wp).extrude(length)
        bbox = beam.obj.BoundBox

        assert beam.obj.isValid()
        assert bbox.XLength == pytest.approx(section.flange_width)
        assert bbox.YLength == pytest.approx(section.depth)
        assert bbox.ZLength == pytest.approx(length)
        assert beam.volume() == pytest.approx(_ipe_area(section) * length)

        fcstd_path = result_dir / "ipe80_beam.FCStd"
        app.to_fcstd(str(fcstd_path), shapes=[beam], feature_name_prefix="IPE80")
        assert fcstd_path.exists()
        assert fcstd_path.stat().st_size > 0

    def test_ipe_beam_joined_frame_can_export_and_reopen(self, app, result_dir):
        FreeCAD = importlib.import_module("FreeCAD")
        from ...rapidcadpy.components import profiles

        section = profiles.ipe("IPE80")
        column_height = 240.0
        beam_span = 180.0

        left_column = section.sketch(
            app.work_plane("XY", origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
        ).extrude(column_height)

        right_column = section.sketch(
            app.work_plane("XY", origin=(beam_span, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
        ).extrude(column_height)

        top_beam = section.sketch(
            app.work_plane(
                origin=(0.0, 0.0, column_height),
                normal=(1.0, 0.0, 0.0),
            )
        ).extrude(beam_span)

        frame = left_column.union(right_column)
        frame = frame.union(top_beam)
        assert frame.obj.isValid()
        assert frame.volume() > _ipe_area(section) * (2.0 * column_height)

        fcstd_path = result_dir / "ipe80_portal_frame.FCStd"
        app.to_fcstd(str(fcstd_path), shapes=[frame], feature_name_prefix="IPEFrame")
        assert fcstd_path.exists()
        assert fcstd_path.stat().st_size > 0

        reopened = FreeCAD.openDocument(str(fcstd_path))
        try:
            assert reopened.Objects
            assert any(obj.TypeId == "Part::Feature" for obj in reopened.Objects)
        finally:
            FreeCAD.closeDocument(reopened.Name)
