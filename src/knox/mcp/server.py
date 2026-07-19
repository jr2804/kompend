"""KNOX MCP server — thin wrapper that registers all tools on FastMCP.

Tool implementations live in :mod:`knox.mcp.tools`.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from knox.mcp.tools import (
    compile_wiki,
    config_get,
    config_set,
    init_wiki,
    memory_add,
    memory_search,
    memory_synthesize,
    parse_document,
    query_wiki,
    wiki_capture,
    wiki_distill,
    wiki_learn,
    wiki_status,
)

#: Module-level FastMCP instance for stdio transport.
server = FastMCP("KNOX")


def _register_all(srv: FastMCP) -> None:
    """Register every KNOX tool on a FastMCP instance."""
    srv.tool(name="parse_document")(parse_document)
    srv.tool(name="compile_wiki")(compile_wiki)
    srv.tool(name="wiki_capture")(wiki_capture)
    srv.tool(name="wiki_learn")(wiki_learn)
    srv.tool(name="memory_add")(memory_add)
    srv.tool(name="memory_search")(memory_search)
    srv.tool(name="memory_synthesize")(memory_synthesize)
    srv.tool(name="init_wiki")(init_wiki)
    srv.tool(name="query_wiki")(query_wiki)
    srv.tool(name="wiki_status")(wiki_status)
    srv.tool(name="wiki_distill")(wiki_distill)
    srv.tool(name="config_get")(config_get)
    srv.tool(name="config_set")(config_set)


# Register on the module-level instance.
_register_all(server)


def main() -> None:
    """Run the KNOX MCP server over stdio transport."""
    server.run(transport="stdio")


def create_mcp_server(port: int = 8000) -> FastMCP:
    """Create a new FastMCP instance with all KNOX tools registered.

    Used by both ``knox mcp`` (stdio) and the ACP agent's companion mode
    (HTTP on the given *port*).
    """
    srv = FastMCP("KNOX", port=port)
    _register_all(srv)
    return srv
