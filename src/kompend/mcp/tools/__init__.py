"""KOMPEND MCP tools — each submodule groups tools by domain.

- :mod:`kompend.mcp.tools.knowledge` — compilation, parsing, capture, distill, write
- :mod:`kompend.mcp.tools.query` — wiki query and status
- :mod:`kompend.mcp.tools.memory` — memory add/search/synthesize
- :mod:`kompend.mcp.tools.config` — init, get, set, list, path, check
- :mod:`kompend.mcp.tools.sources` — list, tag, untag, export
- :mod:`kompend.mcp.tools.references` — list, export, fetch
- :mod:`kompend.mcp.tools.attachments` — add, remove (list/read/info in knowledge)
- :mod:`kompend.mcp.tools.pipeline` — ingest, index, organize, write
- :mod:`kompend.mcp.tools.quality` — lint*, doctor, review, provenance, log, list-parsers
- :mod:`kompend.mcp.tools.packs` — pack-list, pack-install
"""

from kompend.mcp.tools.attachments import (
    attachments_add,
    attachments_adopt,
    attachments_ingest,
    attachments_no_ingest,
    attachments_remove,
    attachments_verify,
)
from kompend.mcp.tools.config import (
    config_check,
    config_get,
    config_list,
    config_path,
    config_set,
    init_wiki,
)
from kompend.mcp.tools.knowledge import (
    compile_wiki,
    parse_document,
    wiki_attachment_info,
    wiki_capture,
    wiki_consolidate,
    wiki_distill,
    wiki_learn,
    wiki_list_attachments,
    wiki_read_attachment,
)
from kompend.mcp.tools.memory import memory_add, memory_search, memory_synthesize
from kompend.mcp.tools.packs import pack_install, pack_list
from kompend.mcp.tools.pipeline import wiki_index, wiki_ingest, wiki_organize, wiki_write
from kompend.mcp.tools.quality import (
    doctor,
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
    provenance,
    review,
)
from kompend.mcp.tools.query import query_wiki, wiki_status
from kompend.mcp.tools.references import (
    references_export,
    references_fetch,
    references_list,
)
from kompend.mcp.tools.sources import (
    sources_attr,
    sources_attr_rm,
    sources_export,
    sources_list,
    sources_tag,
    sources_untag,
)

__all__ = [
    "attachments_add",
    "attachments_adopt",
    "attachments_ingest",
    "attachments_no_ingest",
    "attachments_remove",
    "attachments_verify",
    "compile_wiki",
    "config_check",
    "config_get",
    "config_list",
    "config_path",
    "config_set",
    "doctor",
    "init_wiki",
    "lint_all",
    "lint_check",
    "lint_format",
    "lint_gaps",
    "lint_history",
    "lint_prune",
    "lint_remediate",
    "lint_semantic",
    "list_parsers",
    "log",
    "memory_add",
    "memory_search",
    "memory_synthesize",
    "pack_install",
    "pack_list",
    "parse_document",
    "provenance",
    "query_wiki",
    "references_export",
    "references_fetch",
    "references_list",
    "review",
    "sources_attr",
    "sources_attr_rm",
    "sources_export",
    "sources_list",
    "sources_tag",
    "sources_untag",
    "wiki_attachment_info",
    "wiki_capture",
    "wiki_consolidate",
    "wiki_distill",
    "wiki_index",
    "wiki_ingest",
    "wiki_learn",
    "wiki_list_attachments",
    "wiki_organize",
    "wiki_read_attachment",
    "wiki_status",
    "wiki_write",
]
