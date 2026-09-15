"""KNOX MCP server — thin wrapper that registers all tools on fastmcp (Prefect).

Tool implementations live in :mod:`knox.mcp.tools`. The server uses the
Prefect-maintained ``fastmcp`` framework (built on the official mcp SDK
2.x) — ``FastMCP`` here is ``fastmcp.FastMCP``, not the official SDK's
renamed ``MCPServer`` (memory: keep this import on fastmcp, never pin
mcp<2).
"""

from __future__ import annotations

from fastmcp import FastMCP

from knox.mcp.tools import (
    attachments_add,
    attachments_ingest,
    attachments_no_ingest,
    attachments_remove,
    compile_wiki,
    config_check,
    config_get,
    config_list,
    config_path,
    config_set,
    doctor,
    init_wiki,
    lint_all,
    lint_check,
    lint_format,
    lint_gaps,
    lint_history,
    lint_prune,
    lint_remediate,
    lint_semantic,
    list_parsers,
    log,
    memory_add,
    memory_search,
    memory_synthesize,
    pack_install,
    pack_list,
    parse_document,
    provenance,
    query_wiki,
    references_export,
    references_fetch,
    references_list,
    review,
    sources_attr,
    sources_attr_rm,
    sources_export,
    sources_list,
    sources_tag,
    sources_untag,
    wiki_attachment_info,
    wiki_capture,
    wiki_consolidate,
    wiki_distill,
    wiki_index,
    wiki_ingest,
    wiki_learn,
    wiki_list_attachments,
    wiki_organize,
    wiki_read_attachment,
    wiki_status,
    wiki_write,
)
from knox.mcp.tools._helpers import set_default_wiki_dir

#: Module-level FastMCP instance for stdio transport.
server = FastMCP("KNOX")


def _register_all(srv: FastMCP) -> None:  # noqa: PLR0915 — one registration per tool; long by design
    """Register every KNOX tool on a FastMCP instance."""
    srv.tool(name="wiki_parse")(parse_document)
    srv.tool(name="wiki_compile")(compile_wiki)
    srv.tool(name="wiki_capture")(wiki_capture)
    srv.tool(name="wiki_learn")(wiki_learn)
    srv.tool(name="wiki_write")(wiki_write)
    srv.tool(name="wiki_ingest")(wiki_ingest)
    srv.tool(name="wiki_index")(wiki_index)
    srv.tool(name="wiki_organize")(wiki_organize)
    srv.tool(name="memory_add")(memory_add)
    srv.tool(name="memory_search")(memory_search)
    srv.tool(name="memory_synthesize")(memory_synthesize)
    srv.tool(name="wiki_init")(init_wiki)
    srv.tool(name="wiki_query")(query_wiki)
    srv.tool(name="wiki_status")(wiki_status)
    srv.tool(name="wiki_distill")(wiki_distill)
    srv.tool(name="wiki_consolidate")(wiki_consolidate)
    srv.tool(name="wiki_list_attachments")(wiki_list_attachments)
    srv.tool(name="wiki_read_attachment")(wiki_read_attachment)
    srv.tool(name="wiki_attachment_info")(wiki_attachment_info)
    srv.tool(name="attachments_add")(attachments_add)
    srv.tool(name="attachments_ingest")(attachments_ingest)
    srv.tool(name="attachments_no_ingest")(attachments_no_ingest)
    srv.tool(name="attachments_remove")(attachments_remove)
    srv.tool(name="config_get")(config_get)
    srv.tool(name="config_set")(config_set)
    srv.tool(name="config_list")(config_list)
    srv.tool(name="config_path")(config_path)
    srv.tool(name="config_check")(config_check)
    srv.tool(name="sources_list")(sources_list)
    srv.tool(name="sources_tag")(sources_tag)
    srv.tool(name="sources_untag")(sources_untag)
    srv.tool(name="sources_attr")(sources_attr)
    srv.tool(name="sources_attr_rm")(sources_attr_rm)
    srv.tool(name="sources_export")(sources_export)
    srv.tool(name="references_list")(references_list)
    srv.tool(name="references_export")(references_export)
    srv.tool(name="references_fetch")(references_fetch)
    srv.tool(name="lint_all")(lint_all)
    srv.tool(name="lint_check")(lint_check)
    srv.tool(name="lint_semantic")(lint_semantic)
    srv.tool(name="lint_history")(lint_history)
    srv.tool(name="lint_prune")(lint_prune)
    srv.tool(name="lint_gaps")(lint_gaps)
    srv.tool(name="lint_remediate")(lint_remediate)
    srv.tool(name="lint_format")(lint_format)
    srv.tool(name="doctor")(doctor)
    srv.tool(name="review")(review)
    srv.tool(name="provenance")(provenance)
    srv.tool(name="log")(log)
    srv.tool(name="list_parsers")(list_parsers)
    srv.tool(name="pack_list")(pack_list)
    srv.tool(name="pack_install")(pack_install)


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
    if wiki_dir is not None:
        set_default_wiki_dir(wiki_dir)
    server.run(transport="stdio")


def create_mcp_server() -> FastMCP:
    """Create a new FastMCP instance with all KNOX tools registered.

    Used by both ``knox mcp`` (stdio) and the ACP agent's companion mode
    (HTTP). The port is no longer a constructor argument (fastmcp 4.x) —
    it is passed to ``run_http_async(port=...)`` at the call site.
    """
    srv = FastMCP("KNOX")
    _register_all(srv)
    return srv
