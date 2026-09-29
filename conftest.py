"""Test-suite boundaries for a standalone RapidCADPy checkout."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.resolve()

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_GROUPS = ("core", "ocp", "freecad", "inventor", "fea")

_GROUP_DIRECTORIES = {
    ("tests", "test_fea"): "fea",
    ("tests", "test_integrations", "freecad"): "freecad",
    ("tests", "test_integrations", "inventor"): "inventor",
    ("tests", "test_integrations", "occ"): "ocp",
}

_GROUP_FILES = {
    "tests/test_3d_visualization.py": "ocp",
    "tests/test_app_tracking.py": "ocp",
    "tests/test_freecad_gui_attach.py": "freecad",
    "tests/test_item_components.py": "freecad",
    "tests/test_integrations/test_fillet.py": "freecad",
    "tests/test_integrations/test_loft.py": "freecad",
    "tests/test_integrations/test_profiles.py": "freecad",
    "tests/test_integrations/test_polyline.py": "ocp",
    "tests/test_integrations/test_shape.py": "ocp",
    "tests/test_integrations/test_fluent_api_inventor.py": "inventor",
    "tests/test_integrations/test_inventor_reverse_engineer.py": "inventor",
    "tests/test_integrations/test_inventor_stl_export.py": "inventor",
}

# These tests target APIs that predate the current public package. Keeping the
# boundary explicit prevents stale imports from breaking collection while the
# cases are rewritten against the current contracts.
_LEGACY_DIRECTORIES = {
    ("tests", "test_abstracts"),
    ("tests", "test_importers"),
    ("tests", "test_primitives"),
}
_LEGACY_FILES = {
    "tests/test_integrations/test_cad_modeling.py",
    "tests/test_integrations/test_fea_analyzer.py",
    "tests/test_integrations/test_fluent_api_occ.py",
    "tests/test_integrations/test_occ_shape.py",
    "tests/test_integrations/test_thread.py",
    "tests/test_fea/test_fea_analyzer.py",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("rapidcadpy", "RapidCADPy test selection")
    group.addoption(
        "--test-group",
        action="append",
        choices=(*_GROUPS, "all"),
        default=None,
        help="test group to collect; repeat to select more than one (default: core)",
    )


def _selected_groups(config: pytest.Config) -> set[str]:
    requested = config.getoption("test_group") or ["core"]
    return set(_GROUPS) if "all" in requested else set(requested)


def _relative(path: Path) -> Path | None:
    try:
        return path.resolve().relative_to(_ROOT)
    except ValueError:
        return None


def _test_group(path: Path) -> str | None:
    relative = _relative(path)
    if relative is None or not relative.parts or relative.parts[0] != "tests":
        return None

    posix = relative.as_posix()
    if posix in _LEGACY_FILES:
        return "legacy"

    for prefix in _LEGACY_DIRECTORIES:
        if relative.parts[: len(prefix)] == prefix:
            return "legacy"

    if posix in _GROUP_FILES:
        return _GROUP_FILES[posix]

    for prefix, group in _GROUP_DIRECTORIES.items():
        if relative.parts[: len(prefix)] == prefix:
            return group

    return "core" if path.is_file() else None


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    group = _test_group(collection_path)
    if group == "legacy":
        return True
    return group is not None and group not in _selected_groups(config)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        group = _test_group(Path(str(item.path))) or "core"
        item.add_marker(getattr(pytest.mark, group))
