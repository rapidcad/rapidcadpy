"""Real FreeCAD integration tests isolated from the pytest interpreter."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.parametrize(
    "scenario",
    [
        "reuse_and_classification",
        "update_and_reopen",
        "rollback_creation_and_hydration",
        "rollback_update_and_external_revision",
        "constraints_paths_and_invalid_inputs",
        "planar_spline_sweeps",
        "sweep_corner_modes",
        "spline_inputs_and_rejections",
        "sweep_rollback_and_invalid_modes",
        "loft_alignment_and_order",
        "typed_segments_and_feature_edits",
        "sweep_feature_edits_and_typed_profile_updates",
        "bridge_server_contract_guard",
    ],
)
def test_native_persistent_profile_scenario(scenario, tmp_path):
    executable = Path(
        os.environ.get(
            "FREECAD_PYTHON", "/Applications/FreeCAD.app/Contents/Resources/bin/python"
        )
    )
    library = Path(
        os.environ.get(
            "FREECAD_LIB_PATH", "/Applications/FreeCAD.app/Contents/Resources/lib"
        )
    )
    if not executable.is_file() or not library.is_dir():
        pytest.skip("Requires FreeCAD Python; set FREECAD_PYTHON and FREECAD_LIB_PATH.")
    package_root = Path(__file__).resolve().parents[3]
    script = Path(__file__).with_name("native_profile_scenarios.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(package_root), str(library)])
    result = subprocess.run(
        [str(executable), str(script), scenario, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native scenario passed" in result.stdout
