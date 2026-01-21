import pytest
from rapidcadpy.integrations.occ.app import OpenCascadeApp


class TestShapeVolume:
    @pytest.fixture
    def app(self) -> OpenCascadeApp:
        return OpenCascadeApp()

    def test_shape_volume(self, app: OpenCascadeApp):
        workplane = app.work_plane("XY")
        box = workplane.rect(10, 10).extrude(10)
        volume = box.volume()
        assert abs(volume - 1000.0) < 1e-3, f"Expected volume ~1000.0, got {volume}"
