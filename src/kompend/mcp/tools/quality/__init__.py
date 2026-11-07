"""Quality & diagnostics MCP tools.

Six families, one module each, all re-exported here:

* :mod:`kompend.mcp.tools.quality.lint` - the lint pipeline, its history and gaps (9 tools)
* :mod:`kompend.mcp.tools.quality.doctor` - configuration and connectivity report
* :mod:`kompend.mcp.tools.quality.review` - the review queue
* :mod:`kompend.mcp.tools.quality.provenance` - page-to-source tracing
* :mod:`kompend.mcp.tools.quality.log` - the pipeline log
* :mod:`kompend.mcp.tools.quality.list_parsers` - the ingestion parser

The tool names in ``__all__`` are the MCP contract: ``server.py``
registers each one by name, and ``tests/unit/test_mcp_quality.py`` pins
the set, the input schema and the payload keys.
"""

from kompend.mcp.tools.quality.doctor import doctor
from kompend.mcp.tools.quality.lint import (
    lint_all,
    lint_check,
    lint_format,
    lint_gaps,
    lint_history,
    lint_prune,
    lint_remediate,
    lint_semantic,
    lint_show,
)
from kompend.mcp.tools.quality.list_parsers import list_parsers
from kompend.mcp.tools.quality.log import log
from kompend.mcp.tools.quality.provenance import provenance
from kompend.mcp.tools.quality.review import review

__all__ = [
    "doctor",
    "lint_all",
    "lint_check",
    "lint_format",
    "lint_gaps",
    "lint_history",
    "lint_prune",
    "lint_remediate",
    "lint_semantic",
    "lint_show",
    "list_parsers",
    "log",
    "provenance",
    "review",
]
