"""Named-parameter tests for the OpenCASCADE (pythonOCC) backend."""

import pytest

from ..base.test_parameters import _BaseParameterTests


class TestOccParameters(_BaseParameterTests):
    """Runs the full parameter contract suite against ``OpenCascadeApp``."""

    @pytest.fixture
    def app(self):
        try:
            from rapidcadpy.integrations.occ.app import OpenCascadeApp

            return OpenCascadeApp()
        except Exception as e:
            pytest.skip(f"OCC backend not available: {e}")
