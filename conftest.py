"""
Root conftest for the rapidcadpy test suite.

Inserts vendor/rapidcadpy at the front of sys.path so that the local
development source is always imported — regardless of whether pytest is
invoked from this directory or from the parent workspace root (e.g. via
VS Code's pytest runner which uses --rootdir=<workspace>).
"""

import sys
from pathlib import Path

# Absolute path to vendor/rapidcadpy/  (the directory containing this file)
_HERE = Path(__file__).parent.resolve()

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
