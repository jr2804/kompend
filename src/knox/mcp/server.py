"""KNOX MCP server — thin wrapper that registers all tools on FastMCP.

Tool implementations live in :mod:`knox.mcp.tools`.
"""

from __future__ import annotations

import os

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

#: Server-wide default wiki directory, set by :func:`main`.
_DEFAULT_WIKI_DIR: str | None = None


def resolve_wiki_dir(wiki_dir: str | None = None) -> str:
    """Resolve the wiki directory from most-specific source.

    Priority: explicit argument > server-wide default (set via
    ``knox mcp --wiki-dir``) > ``KNOX_WIKI_DIR`` env var > current directory.
    """
    if wiki_dir is not None:
        return wiki_dir
    if _DEFAULT_WIKI_DIR is not None:
        return _DEFAULT_WIKI_DIR
    return os.environ.get("KNOX_WIKI_DIR", ".")


def _register_all(srv: FastMCP) -> None:
    """Register every KNOX tool on a FastMCP instance."""
    srv.tool(name="wiki_parse")(parse_document)
    srv.tool(name="wiki_compile")(compile_wiki)
    srv.tool(name="wiki_capture")(wiki_capture)
    srv.tool(name="wiki_learn")(wiki_learn)
    srv.tool(name="memory_add")(memory_add)
    srv.tool(name="memory_search")(memory_search)
    srv.tool(name="memory_synthesize")(memory_synthesize)
    srv.tool(name="wiki_init")(init_wiki)
    srv.tool(name="wiki_query")(query_wiki)
    srv.tool(name="wiki_status")(wiki_status)
    srv.tool(name="wiki_distill")(wiki_distill)
    srv.tool(name="config_get")(config_get)
    srv.tool(name="config_set")(config_set)


# Register on the module-level instance.
_register_all(server)


def main(wiki_dir: str | None = None) -> None:
    """Run the KNOX MCP server over stdio transport.

    Parameters
    ----------
    wiki_dir:
        Server-wide default wiki directory. When set, every tool that
        accepts a ``wiki_dir`` parameter will use this value unless an
        explicit override is provided in the tool call. Also settable via
        the ``KNOX_WIKI_DIR`` environment variable.
    """
    global _DEFAULT_WIKI_DIR
    if wiki_dir is not None:
        _DEFAULT_WIKI_DIR = wiki_dir
    server.run(transport="stdio")


def create_mcp_server(port: int = 8000) -> FastMCP:
    """Create a new FastMCP instance with all KNOX tools registered.

    Used by both ``knox mcp`` (stdio) and the ACP agent's companion mode
    (HTTP on the given *port*).
    """
    srv = FastMCP("KNOX", port=port)
    _register_all(srv)
    return srv
