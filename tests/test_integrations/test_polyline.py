import math

import pytest

from vendor.rapidcadpy.rapidcadpy.integrations.freecad import FreeCADApp
from vendor.rapidcadpy.rapidcadpy.integrations.ocp.app import OpenCascadeOcpApp


class TestFillet:
    """Reusable fillet tests for backend-specific integration suites.

    Concrete test modules should inherit from this class and provide an
    ``app`` fixture that returns the backend-specific app instance.
    """

    @pytest.fixture
    def app(self):
        # app = FreeCADApp()
        app = OpenCascadeOcpApp()
        yield app

    def test_polyline_pipe_creates_solid(self, app):
        spine = app.sketch_3d.move_to(0.0, 0.0, 0.0).polyline(
            [
                (0.0, 0.0, 0.0),
                (3.0, 4.0, 12.0),
            ]
        )

        pipe = spine.pipe(diameter=2.0, is_frenet=True, transition_mode="right")

        assert pipe is not None
        assert pipe.volume() > 0

        expected_length = math.sqrt(3.0**2 + 4.0**2 + 12.0**2)
        expected_volume = math.pi * (1.0**2) * expected_length
        assert pipe.volume() == pytest.approx(expected_volume, rel=0.1)
