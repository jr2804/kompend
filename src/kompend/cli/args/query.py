"""CLI arguments for the query command."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from cyclopts import Parameter

from kompend.cli.args.common import in_range
from kompend.core.vocab import LlmDataFormat, Verbosity

# ── Q17 query command ──────────────────────────────────────────────────

QueryText = Annotated[
    str,
    Parameter(
        help="Natural language question to ask against the compiled wiki.",
    ),
]

QueryTopK = Annotated[
    int,
    Parameter(
        name="--top-k",
        help="Number of context chunks to retrieve (default: 5).",
        validator=in_range(1, 50),
    ),
]

QueryRaw = Annotated[
    bool,
    Parameter(
        name="--raw",
        help="Return raw context chunks without LLM synthesis.",
        negative="",
    ),
]

QueryFormatOption = Annotated[
    LlmDataFormat | None,
    Parameter(
        name="--format",
        help="Output format for agents: markdown (default), json, toon, ison. Defaults to query.format from config.",
    ),
]

QueryNoClassify = Annotated[
    bool,
    Parameter(
        name="--no-classify",
        help="Skip intent classification (always retrieve).",
        negative="",
    ),
]


class SearchMode(StrEnum):
    WIKI = "wiki"
    RAW = "raw"
    HYBRID = "hybrid"
    GRAPH = "graph"


QuerySearchMode = Annotated[
    SearchMode,
    Parameter(
        name=("--search-mode", "--scope"),
        help="Search mode: wiki (compiled pages), raw (source files), hybrid (both), or graph (hybrid + graph traversal expansion).",
    ),
]

QueryAdaptive = Annotated[
    bool,
    Parameter(
        name="--adaptive",
        help="Use adaptive retrieval (context-budget-aware zoom-in/zoom-out/targeted).",
        negative="",
    ),
]

QueryRemember = Annotated[
    bool,
    Parameter(
        name="--remember",
        help="Save this query and synthesized answer to the memory store.",
        negative="",
    ),
]

QueryVerbosity = Annotated[
    Verbosity | None,
    Parameter(
        name="--verbosity",
        help="Answer verbosity: brief, standard, or comprehensive. Defaults to search.synthesis.verbosity from config.",
    ),
]
