# RapidCAD.py Docs MCP Server

A [FastMCP](https://github.com/jlowin/fastmcp) server that exposes the
RapidCAD.py documentation as MCP resources and tools.

## Exposed API

| Name | Type | Description |
|---|---|---|
| `docs://index` | Resource | Top-level documentation index |
| `list_docs` | Tool | List all doc pages (path, title, description, section) |
| `get_doc` | Tool | Return the full content of a page by path |
| `search_docs` | Tool | Full-text keyword search across all pages |
| `get_doc_section` | Tool | Return all pages belonging to a section |

### Sections

| Section slug | Content |
|---|---|
| `root` | Top-level index |
| `computer-aided-design` | Fluent API, shapes, workplanes, sweep, profiles |
| `finite-element-analysis` | FEA analysis, constraints, loads, materials, meshing, optimization, visualization |
| `advanced` | 2D/3D exports, visualizations, Inventor reverse-engineering |
| `api` | Low-level mesher example |

## Running

### Standalone (stdio transport – default)

```bash
# from the repo root
python vendor/rapidcadpy/mcp/server.py
```

Or via the FastMCP CLI:

```bash
fastmcp run vendor/rapidcadpy/mcp/server.py
```

### HTTP / SSE transport

```bash
fastmcp run vendor/rapidcadpy/mcp/server.py --transport sse --port 8765
```

## Adding to Claude Desktop / VS Code Copilot

Add an entry to your MCP client config:

```json
{
  "mcpServers": {
    "rapidcadpy-docs": {
      "command": "python",
      "args": ["vendor/rapidcadpy/mcp/server.py"]
    }
  }
}
```

## Requirements

```
fastmcp>=0.1.0
```

`fastmcp` is not yet in `requirements.txt` – install with:

```bash
pip install fastmcp
```
