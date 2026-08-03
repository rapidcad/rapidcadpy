"""Shape volume + cross-kernel boolean parity tests.

Environment reality (why this file is structured the way it is):

* The FreeCAD backend can only be imported by FreeCAD's bundled Python
  (ABI-locked ``FreeCAD.so``). The OpenCascade (OCP) backend can only be
  imported by the project's 3.12 ``ocp`` env. The two working kernels therefore
  live in *different interpreters* and can never share a process.

* So per-backend tests (``TestShapeVolume``) use an in-process fixture that
  skips when its backend isn't importable in the current interpreter — run the
  file once under each interpreter to cover both.

* Cross-kernel parity (``TestCrossKernelParity``) can't be in-process. It runs
  each geometry script in the correct interpreter via a subprocess and compares
  the resulting volumes, which is what actually proves the booleans agree.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# Paths / interpreter discovery
# ---------------------------------------------------------------------------

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_VENDOR_RC = os.path.join(_ROOT, "vendor", "rapidcadpy")

_FREECAD_LIB_CANDIDATES = [
    os.environ.get("FREECAD_LIB_PATH", "").strip(),
    "/Applications/FreeCAD.app/Contents/Resources/lib",
    "/usr/lib/freecad/lib",
    "/usr/lib64/freecad/lib",
]
_FREECAD_PY_CANDIDATES = [
    os.environ.get("FREECAD_PYTHON", "").strip(),
    "/Applications/FreeCAD.app/Contents/Resources/bin/python",
]
_OCP_PY_CANDIDATES = [
    "/opt/homebrew/Caskroom/miniconda/base/envs/ocp/bin/python",
    "/opt/conda/envs/ocp/bin/python",
]


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _freecad_lib_path():
    return _first_existing(_FREECAD_LIB_CANDIDATES)


def _interpreter_for(backend: str):
    """Return a Python interpreter able to import *backend*, or None."""
    if backend == "freecad":
        return _first_existing(_FREECAD_PY_CANDIDATES)
    if backend == "ocp":
        if importlib.util.find_spec("OCP") is not None:
            return sys.executable
        return _first_existing(_OCP_PY_CANDIDATES)
    raise ValueError(f"unknown backend: {backend}")


# ---------------------------------------------------------------------------
# In-process per-backend fixture
# ---------------------------------------------------------------------------


def _make_app(backend: str):
    if backend == "freecad":
        from vendor.rapidcadpy.rapidcadpy.integrations.freecad import FreeCADApp

        return FreeCADApp()
    if backend == "ocp":
        from vendor.rapidcadpy.rapidcadpy import OpenCascadeOcpApp

        return OpenCascadeOcpApp()
    raise ValueError(f"unknown backend: {backend}")


@pytest.fixture(params=["freecad", "ocp"])
def app(request):
    backend = request.param
    try:
        return _make_app(backend)
    except Exception as exc:  # ImportError / ABI / missing kernel
        pytest.skip(f"{backend} backend not importable in this interpreter: {exc}")


class TestShapeVolume:
    def test_shape_volume(self, app):
        workplane = app.work_plane("XY")
        box = workplane.rect(10, 10).extrude(10)
        volume = box.volume()
        assert abs(volume - 1000.0) < 1e-3, f"Expected volume ~1000.0, got {volume}"

    def test_union_volume(self, app):
        """Volume after a boolean union of two overlapping boxes."""
        box1 = app.work_plane("XY").rect(10, 10, centered=True).close().extrude(5)
        volume1 = box1.volume()

        box2 = (
            app.work_plane("XY")
            .move_to(5, 0)
            .rect(10, 10, centered=True)
            .close()
            .extrude(5)
        )
        volume2 = box2.volume()

        result = box1.union(box2)
        result_volume = result.volume()

        assert result_volume > 0, "Union volume should be positive"
        assert result_volume < (volume1 + volume2), "Union should subtract overlap"
        assert result_volume > max(volume1, volume2), "Union >= either operand"

    def test_union_multiple_shapes(self, app):
        """Union with a list of multiple shapes."""
        base = app.work_plane("XY").rect(10, 10, centered=True).close().extrude(5)
        base_volume = base.volume()

        shapes_to_union = []
        for i in range(3):
            box = (
                app.work_plane("XY")
                .move_to(5 + i * 5, 0)
                .rect(5, 5, centered=True)
                .close()
                .extrude(5)
            )
            shapes_to_union.append(box)

        result = base.union(shapes_to_union)
        result_volume = result.volume()

        assert result_volume > base_volume, "Union should increase volume"
        assert result_volume > 0, "Result volume should be positive"


# ---------------------------------------------------------------------------
# Cross-kernel parity (subprocess per backend)
# ---------------------------------------------------------------------------

# Each case builds a final solid named ``result`` from the injected ``app``.
# Keep them backend-agnostic (no app instantiation, no kernel-specific calls).
PARITY_CASES = {
    "box": "result = app.work_plane('XY').rect(10, 10, centered=True).close().extrude(10)",
    "union_overlap": (
        "a = app.work_plane('XY').rect(10, 10, centered=True).close().extrude(5)\n"
        "b = app.work_plane('XY').move_to(5, 0).rect(10, 10, centered=True).close().extrude(5)\n"
        "result = a.union(b)"
    ),
    "cut": (
        "a = app.work_plane('XY').rect(20, 20, centered=True).close().extrude(10)\n"
        "b = app.work_plane('XY').rect(10, 10, centered=True).close().extrude(10)\n"
        "result = a.cut(b)"
    ),
    # Offset workplanes — suspected source of cross-kernel divergence.
    "offset_plane_xz": "result = app.work_plane('XZ', offset=25).rect(10, 10, centered=True).close().extrude(4)",
    "offset_plane_yz": "result = app.work_plane('YZ', offset=25).rect(10, 10, centered=True).close().extrude(4)",
    # Isolate the chord member alone (XZ offset + move_to + centered=False).
    "chord_xz_offset": "result = app.work_plane('XZ', offset=10).move_to(0, 0).rect(100, 10, centered=False).extrude(6)",
    "deck_xy": "result = app.work_plane('XY').move_to(0, 0).rect(100, 20, centered=False).extrude(8)",
    # Deck + chord on an offset plane, then union (mirrors the bridge pattern).
    "union_offset_members": (
        "deck = app.work_plane('XY').move_to(0, 0).rect(100, 20, centered=False).extrude(8)\n"
        "chord = app.work_plane('XZ', offset=10).move_to(0, 0).rect(100, 10, centered=False).extrude(6)\n"
        "result = deck.union(chord)"
    ),
}

_RUNNER_TEMPLATE = """\
import json, os, sys
for p in {paths!r}:
    if p and p not in sys.path:
        sys.path.insert(0, p)
freecad_lib = {freecad_lib!r}
if freecad_lib:
    os.environ.setdefault("FREECAD_LIB_PATH", freecad_lib)
    if freecad_lib not in sys.path:
        sys.path.insert(0, freecad_lib)

if {backend!r} == "freecad":
    from vendor.rapidcadpy.rapidcadpy.integrations.freecad import FreeCADApp
    app = FreeCADApp()
else:
    from vendor.rapidcadpy.rapidcadpy import OpenCascadeOcpApp
    app = OpenCascadeOcpApp()

ns = {{"app": app}}
exec({case!r}, ns)
result = ns["result"]
volume = float(result.volume())

shape = result.obj
if {backend!r} == "freecad":
    bb = shape.BoundBox
    bbox = [bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax]
else:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    _bb = Bnd_Box()
    BRepBndLib.Add_s(shape, _bb)
    xmin, ymin, zmin, xmax, ymax, zmax = _bb.Get()
    bbox = [xmin, ymin, zmin, xmax, ymax, zmax]

print("PARITY_JSON:" + json.dumps({{"volume": volume, "bbox": [float(v) for v in bbox]}}))
"""


def _run_case_in_backend(backend: str, case_src: str):
    """Execute one geometry case in *backend*'s interpreter; return metrics dict.

    Returns None when no interpreter for the backend is available (caller skips).
    """
    interpreter = _interpreter_for(backend)
    if interpreter is None:
        return None

    program = _RUNNER_TEMPLATE.format(
        paths=[_ROOT, _VENDOR_RC],
        freecad_lib=_freecad_lib_path() or "",
        backend=backend,
        case=case_src,
    )
    proc = subprocess.run(
        [interpreter, "-c", program],
        capture_output=True,
        text=True,
        timeout=180,
        stdin=subprocess.DEVNULL,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("PARITY_JSON:"):
            return json.loads(line[len("PARITY_JSON:") :])
    raise RuntimeError(
        f"{backend} runner produced no result (exit {proc.returncode}).\n"
        f"stdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-1000:]}"
    )


class TestCrossKernelParity:
    """The same backend-agnostic script must yield the same volume in every kernel."""

    @pytest.mark.parametrize("case_name", list(PARITY_CASES))
    def test_volume_matches_across_kernels(self, case_name):
        case_src = PARITY_CASES[case_name]

        freecad = _run_case_in_backend("freecad", case_src)
        ocp = _run_case_in_backend("ocp", case_src)

        if freecad is None or ocp is None:
            missing = "freecad" if freecad is None else "ocp"
            pytest.skip(f"no interpreter available for the {missing} backend")

        v_fc = freecad["volume"]
        v_ocp = ocp["volume"]
        rel = abs(v_fc - v_ocp) / max(abs(v_ocp), 1e-9)
        assert rel < 1e-3, (
            f"[{case_name}] volume differs across kernels: "
            f"freecad={v_fc:.4f}, ocp={v_ocp:.4f} (rel diff {rel:.3%})"
        )

        # Position parity: a translated solid has identical volume, so compare
        # the bounding box too — this is what catches offset-plane placement bugs.
        bb_fc = freecad["bbox"]
        bb_ocp = ocp["bbox"]
        worst = max(abs(a - b) for a, b in zip(bb_fc, bb_ocp))
        assert worst < 1e-3, (
            f"[{case_name}] bounding box differs across kernels (placement):\n"
            f"  freecad={['%.3f' % v for v in bb_fc]}\n"
            f"  ocp    ={['%.3f' % v for v in bb_ocp]}\n"
            f"  worst coord delta={worst:.3f}"
        )
