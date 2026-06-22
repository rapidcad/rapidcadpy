import math
import importlib

import pytest

from vendor.rapidcadpy.rapidcadpy.integrations.freecad import FreeCADApp


class TestFillet:
    """Reusable fillet tests for backend-specific integration suites.

    Concrete test modules should inherit from this class and provide an
    ``app`` fixture that returns the backend-specific app instance.
    """

    @pytest.fixture
    def app(self):
        app = FreeCADApp()
        yield app

    def test_fillet_parallel_z_edges(self, app):
        workplane = app.work_plane("XY")
        box = workplane.rect(3.0, 3.0).close().extrude(0.5)

        original_volume = box.volume()
        fillet_radius = 0.125

        result = box.edges("|Z").fillet(fillet_radius)

        expected_removed_volume = 0.5 * (4.0 - math.pi) * (fillet_radius**2)
        expected_volume = original_volume - expected_removed_volume

        assert result is box
        assert result.volume() < original_volume
        assert result.volume() > 0.0
        assert result.volume() == pytest.approx(expected_volume, abs=1e-3)

    def test_filleted_shape_can_export_to_step(self, app, tmp_path):
        workplane = app.work_plane("XY")
        box = workplane.rect(3.0, 3.0).close().extrude(0.5)

        filleted = box.edges("|Z").fillet(0.125)
        step_path = tmp_path / "filleted_box.step"
        print(f"Exporting filleted shape to STEP: {step_path}")
        filleted.to_step(str(step_path))
        assert step_path.exists()
        assert step_path.stat().st_size > 0

        with step_path.open("r", encoding="utf-8", errors="ignore") as step_file:
            assert step_file.readline().strip() == "ISO-10303-21;"

    def test_fillet_save_to_fcstd(self, app, tmp_path):
        FreeCAD = importlib.import_module("FreeCAD")

        workplane = app.work_plane("XY")
        box = workplane.rect(3.0, 3.0).close().extrude(0.5)
        filleted = box.edges("|Z").fillet(0.125)

        fcstd_path = tmp_path / "filleted_box.FCStd"
        print(f"Exporting filleted shape to FreeCAD file: {fcstd_path}")
        app.to_fcstd(
            str(fcstd_path),
            shapes=[filleted],
            feature_name_prefix="Result",
        )
        filleted.to_fcstd("outputs/filleted_box.FCStd")
        assert fcstd_path.exists()
        assert fcstd_path.stat().st_size > 0

        reopened = FreeCAD.openDocument(str(fcstd_path))
        try:
            object_types = [obj.TypeId for obj in reopened.Objects]
            assert "Sketcher::SketchObject" in object_types
            assert "Part::Extrusion" in object_types
            assert "Part::Feature" in object_types
        finally:
            FreeCAD.closeDocument(reopened.Name)
