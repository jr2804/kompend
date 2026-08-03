"""KNOX MCP tools — each submodule groups tools by domain.

- :mod:`knox.mcp.tools.knowledge` — compilation, parsing, capture, distill
- :mod:`knox.mcp.tools.query` — wiki query and status
- :mod:`knox.mcp.tools.memory` — memory add/search/synthesize
- :mod:`knox.mcp.tools.config` — init, get, set
"""

from knox.mcp.tools.config import config_get, config_set, init_wiki
from knox.mcp.tools.knowledge import (
    compile_wiki,
    parse_document,
    wiki_asset_info,
    wiki_capture,
    wiki_distill,
    wiki_learn,
    wiki_list_assets,
    wiki_read_asset,
)
from knox.mcp.tools.memory import memory_add, memory_search, memory_synthesize
from knox.mcp.tools.query import query_wiki, wiki_status

__all__ = [
    "compile_wiki",
    "config_get",
    "config_set",
    "init_wiki",
    "memory_add",
    "memory_search",
    "memory_synthesize",
    "parse_document",
    "query_wiki",
    "wiki_asset_info",
    "wiki_capture",
    "wiki_distill",
    "wiki_learn",
    "wiki_list_assets",
    "wiki_read_asset",
    "wiki_status",
]
