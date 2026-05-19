"""Small localhost download server for RapidCADPy MCP exports."""

from __future__ import annotations

import mimetypes
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import quote


_SERVER: Optional[ThreadingHTTPServer] = None
_SERVER_THREAD: Optional[threading.Thread] = None
_SERVER_ROOT: Optional[Path] = None
_SERVER_URL: Optional[str] = None


class _DownloadHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format: str, *args) -> None:
        # Keep MCP stdio/stderr quiet.
        return

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def guess_type(self, path: str) -> str:
        lower = path.lower()
        if lower.endswith((".step", ".stp")):
            return "model/step"
        if lower.endswith(".fcstd"):
            return "application/vnd.freecad"
        return super().guess_type(path)


def get_export_dir() -> Path:
    root = os.environ.get("RAPIDCADPY_MCP_EXPORT_DIR", "/tmp/rapidcadpy_exports")
    path = Path(root).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_download_server() -> dict:
    """Start/reuse a localhost static server for exported CAD files."""
    global _SERVER, _SERVER_THREAD, _SERVER_ROOT, _SERVER_URL

    root = get_export_dir()
    host = os.environ.get("RAPIDCADPY_MCP_DOWNLOAD_HOST", "127.0.0.1")
    port = int(os.environ.get("RAPIDCADPY_MCP_DOWNLOAD_PORT", "8766"))

    if _SERVER is not None and _SERVER_ROOT == root:
        return {"root": str(_SERVER_ROOT), "base_url": _SERVER_URL}

    handler = lambda *args, **kwargs: _DownloadHandler(
        *args, directory=str(root), **kwargs
    )
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError:
        server = ThreadingHTTPServer((host, 0), handler)

    actual_host, actual_port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    _SERVER = server
    _SERVER_THREAD = thread
    _SERVER_ROOT = root
    _SERVER_URL = f"http://{actual_host}:{actual_port}"
    return {"root": str(root), "base_url": _SERVER_URL}


def export_download_info(path: str) -> dict:
    """Return localhost URL metadata for an exported file path."""
    try:
        server = ensure_download_server()
    except OSError as exc:
        return {
            "download_url": None,
            "download_error": f"Could not start localhost download server: {exc}",
            "export_dir": str(get_export_dir()),
            "filename": Path(path).expanduser().name,
        }
    root = Path(server["root"]).resolve()
    file_path = Path(path).expanduser().resolve()

    try:
        relative = file_path.relative_to(root)
    except ValueError:
        return {
            "download_url": None,
            "download_error": f"File is outside export dir {root}",
            "download_base_url": server["base_url"],
            "export_dir": str(root),
        }

    url_path = "/".join(quote(part) for part in relative.parts)
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if file_path.suffix.lower() in {".step", ".stp"}:
        mime_type = "model/step"
    elif file_path.suffix.lower() == ".fcstd":
        mime_type = "application/vnd.freecad"

    return {
        "download_url": f"{server['base_url']}/{url_path}",
        "download_base_url": server["base_url"],
        "export_dir": str(root),
        "filename": file_path.name,
        "mime_type": mime_type or "application/octet-stream",
    }
