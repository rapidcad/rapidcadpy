"""Named-parameter tests for the OCP (build123d-style) backend."""

import pytest

from ..base.test_parameters import _BaseParameterTests


class TestOcpParameters(_BaseParameterTests):
    """Runs the full parameter contract suite against ``OpenCascadeOcpApp``."""

    @pytest.fixture
    def app(self):
        try:
            from rapidcadpy.integrations.ocp.app import OpenCascadeOcpApp

            return OpenCascadeOcpApp()
        except Exception as e:
            pytest.skip(f"OCP backend not available: {e}")
