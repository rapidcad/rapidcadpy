"""Tests for the RapidCADPy MCP server and live CAD session."""

from __future__ import annotations

import importlib.util
import asyncio
import os
import sys
from pathlib import Path

import pytest

from rapidcadpy.mcp_session import RapidCADSession


SERVER_PATH = Path(__file__).resolve().parents[1] / "mcp" / "server.py"
VENDOR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREECAD_PYTHON = "/Applications/FreeCAD.app/Contents/Resources/bin/python"
DEFAULT_FREECAD_LIB_PATH = "/Applications/FreeCAD.app/Contents/Resources/lib"


def _import_fastmcp_without_local_mcp_shadow():
    """Import fastmcp without vendor/rapidcadpy/mcp shadowing package mcp."""
    vendor_root = str(VENDOR_ROOT)
    old_sys_path = list(sys.path)
    sys.path = [
        entry for entry in sys.path if str(Path(entry or ".").resolve()) != vendor_root
    ]
    try:
        return pytest.importorskip("fastmcp")
    finally:
        sys.path = old_sys_path


def _load_server_module():
    _import_fastmcp_without_local_mcp_shadow()
    spec = importlib.util.spec_from_file_location(
        "rapidcadpy_mcp_server_test", SERVER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_session_reports_missing_backend_before_setup():
    session = RapidCADSession()

    result = session.rect(10, 20)

    assert result["ok"] is False
    assert "setup_backend" in result["error"]


def test_mcp_server_exposes_docs_and_live_cad_tools():
    fastmcp = _import_fastmcp_without_local_mcp_shadow()
    module = _load_server_module()

    async def run_client():
        async with fastmcp.Client(module.mcp) as client:
            return await client.list_tools()

    tools = asyncio.run(run_client())

    tool_names = {tool.name for tool in tools}
    assert {
        "list_docs",
        "get_doc",
        "setup_backend",
        "work_plane",
        "rect",
        "extrude",
        "export_step",
        "download_server_info",
        "describe_freecad_file",
        "describe_state",
    }.issubset(tool_names)


def test_mcp_freecad_worker_smoke(tmp_path, monkeypatch):
    fastmcp = _import_fastmcp_without_local_mcp_shadow()
    freecad_python = os.environ.get("FREECAD_PYTHON", DEFAULT_FREECAD_PYTHON)
    freecad_lib_path = os.environ.get("FREECAD_LIB_PATH", DEFAULT_FREECAD_LIB_PATH)
    if not Path(freecad_python).exists():
        pytest.skip(f"FreeCAD Python not found: {freecad_python}")
    if not Path(freecad_lib_path).exists():
        pytest.skip(f"FreeCAD lib path not found: {freecad_lib_path}")

    worker_log = tmp_path / "freecad_worker.log"
    monkeypatch.setenv("FREECAD_PYTHON", freecad_python)
    monkeypatch.setenv("FREECAD_LIB_PATH", freecad_lib_path)
    monkeypatch.setenv("RAPIDCADPY_MCP_WORKER_LOG", str(worker_log))
    monkeypatch.setenv("RAPIDCADPY_MCP_WORKER_TIMEOUT", "30")

    module = _load_server_module()
    step_path = tmp_path / "mcp_block.step"
    native_path = tmp_path / "mcp_block.FCStd"

    async def run_client():
        async with fastmcp.Client(module.mcp) as client:
            calls = [
                (
                    "setup_backend",
                    {"cad_system": "freecad", "document_name": "mcp_test"},
                ),
                ("work_plane", {"plane": "XY"}),
                ("rect", {"width": 10, "height": 20}),
                ("extrude", {"distance": 5}),
                ("export_native", {"path": str(native_path)}),
                ("export_step", {"path": str(step_path)}),
                ("describe_freecad_file", {"path": str(native_path)}),
                ("describe_state", {}),
            ]
            results = []
            for tool_name, args in calls:
                result = await client.call_tool(tool_name, args)
                results.append(result.structured_content)
            return results

    results = asyncio.run(run_client())

    for result in results:
        assert result["ok"] is True, result
    assert step_path.exists()
    assert native_path.exists()
    assert "download_url" in results[5]
    assert "export_dir" in results[5]
    assert results[6]["document"]["file_name"] == str(native_path)
    assert len(results[6]["objects"]) >= 1
    assert any("geometry" in obj for obj in results[6]["objects"])
    assert results[-1]["active_shape_id"] == "shape_1"
    assert results[-1]["objects"][-1]["geometry"]["volume"] == pytest.approx(1000.0)
