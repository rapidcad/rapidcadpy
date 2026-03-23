"""
RapidCAD.py docs MCP server.

Exposes the documentation under docs/content/docs as resources and tools
so that MCP clients (e.g. Claude, Copilot) can discover and read them.

Run locally:
    python vendor/rapidcadpy/mcp/server.py

Or via the MCP CLI:
    fastmcp run vendor/rapidcadpy/mcp/server.py
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_DOCS_ROOT = _HERE.parent / "docs" / "content" / "docs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RE_FRONTMATTER = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_RE_IMPORT = re.compile(r"^import\s+.*?;\s*$", re.MULTILINE)
# Strip JSX-style component tags while keeping their text content
_RE_JSX_OPEN_CLOSE = re.compile(r"<(\w[\w.]*)[^>]*>(.*?)</\1>", re.DOTALL)
_RE_JSX_SELF_CLOSE = re.compile(r"<\w[\w.]*[^>]*/\s*>")
_RE_JSX_OPEN_ONLY = re.compile(r"</?[\w][\w.]*[^>]*>")


def _strip_mdx(raw: str) -> str:
    """Return the MDX source with frontmatter and JSX stripped."""
    text = _RE_FRONTMATTER.sub("", raw, count=1)
    text = _RE_IMPORT.sub("", text)
    # Unwrap JSX components keeping their content (repeated until stable)
    prev = None
    while prev != text:
        prev = text
        text = _RE_JSX_OPEN_CLOSE.sub(lambda m: m.group(2), text)
    text = _RE_JSX_SELF_CLOSE.sub("", text)
    text = _RE_JSX_OPEN_ONLY.sub("", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_frontmatter(raw: str) -> dict:
    m = _RE_FRONTMATTER.match(raw)
    if not m:
        return {}
    block = m.group(0).strip().strip("-").strip()
    result: dict = {}
    for line in block.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            value = v.strip().strip('"').strip("'")
            result[k.strip()] = value
    return result


def _all_doc_files() -> list[Path]:
    return sorted(_DOCS_ROOT.rglob("*.mdx"))


def _rel(path: Path) -> str:
    """Return a short slash-separated doc path relative to DOCS_ROOT."""
    return path.relative_to(_DOCS_ROOT).with_suffix("").as_posix()


def _meta_for_dir(directory: Path) -> dict:
    """Read meta.json for a directory if present."""
    meta_path = directory / "meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="rapidcadpy-docs",
    instructions=(
        "Provides access to RapidCAD.py documentation covering parametric CAD "
        "modeling, FEA analysis, constraints, loads, materials, meshing, "
        "optimization, export formats, and visualization. "
        "Use list_docs to discover available pages, get_doc to read a page, "
        "and search_docs to find pages by keyword."
    ),
)


# ---------------------------------------------------------------------------
# Resources  – each doc page is a resource at  docs://<rel-path>
# ---------------------------------------------------------------------------

@mcp.resource("docs://index")
def docs_index() -> str:
    """The top-level documentation index."""
    index_path = _DOCS_ROOT / "index.mdx"
    if index_path.exists():
        return _strip_mdx(index_path.read_text())
    return "No index found."


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_docs() -> list[dict]:
    """
    List all available RapidCAD.py documentation pages.

    Returns a list of objects with:
    - path: the doc path to pass to get_doc()
    - title: page title extracted from frontmatter
    - description: short description from frontmatter
    - section: top-level section (e.g. 'computer-aided-design', 'finite-element-analysis')
    """
    result = []
    for f in _all_doc_files():
        raw = f.read_text()
        fm = _parse_frontmatter(raw)
        rel = _rel(f)
        parts = rel.split("/")
        section = parts[0] if len(parts) > 1 else "root"
        result.append(
            {
                "path": rel,
                "title": fm.get("title", parts[-1].replace("-", " ").title()),
                "description": fm.get("description", ""),
                "section": section,
            }
        )
    return result


@mcp.tool()
def get_doc(path: str) -> str:
    """
    Return the full content of a documentation page.

    Args:
        path: The doc path as returned by list_docs(), e.g.
              'finite-element-analysis/fea-analysis'
              'computer-aided-design/fluent-api'
              'index'
    """
    # Normalise: strip leading slash, ensure no .mdx suffix
    path = path.lstrip("/").removesuffix(".mdx")

    candidate = _DOCS_ROOT / (path + ".mdx")
    if not candidate.exists():
        # Maybe they passed a directory slug — try index.mdx inside it
        candidate = _DOCS_ROOT / path / "index.mdx"
    if not candidate.exists():
        available = [_rel(f) for f in _all_doc_files()]
        return (
            f"Doc '{path}' not found.\n\nAvailable paths:\n"
            + "\n".join(f"  - {p}" for p in available)
        )
    return _strip_mdx(candidate.read_text())


@mcp.tool()
def search_docs(query: str, max_results: int = 10) -> list[dict]:
    """
    Full-text search across all documentation pages.

    Args:
        query: keyword or phrase to search for (case-insensitive)
        max_results: maximum number of results to return (default 10)

    Returns a list of matching pages with a short snippet around the match.
    """
    query_lower = query.lower()
    results = []
    for f in _all_doc_files():
        raw = f.read_text()
        content = _strip_mdx(raw)
        if query_lower not in content.lower():
            continue
        fm = _parse_frontmatter(raw)
        rel = _rel(f)

        # Build a short snippet around the first occurrence
        idx = content.lower().find(query_lower)
        start = max(0, idx - 120)
        end = min(len(content), idx + 120)
        snippet = ("..." if start > 0 else "") + content[start:end].strip() + ("..." if end < len(content) else "")

        results.append(
            {
                "path": rel,
                "title": fm.get("title", rel.split("/")[-1].replace("-", " ").title()),
                "snippet": snippet,
            }
        )
        if len(results) >= max_results:
            break
    return results


@mcp.tool()
def get_doc_section(section: str) -> list[dict]:
    """
    Return all pages that belong to a documentation section.

    Args:
        section: one of 'computer-aided-design', 'finite-element-analysis',
                 'advanced', 'api', or 'root'

    Returns a list of {path, title, content} objects.
    """
    pages = []
    for f in _all_doc_files():
        rel = _rel(f)
        parts = rel.split("/")
        page_section = parts[0] if len(parts) > 1 else "root"
        if page_section.lower() != section.lower():
            continue
        raw = f.read_text()
        fm = _parse_frontmatter(raw)
        pages.append(
            {
                "path": rel,
                "title": fm.get("title", parts[-1].replace("-", " ").title()),
                "content": _strip_mdx(raw),
            }
        )
    return pages


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
