"""CLI arguments for the query command."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

import typer

from knox.core.config._configs import LlmDataFormat, Verbosity

# ── Q17 query command ──────────────────────────────────────────────────

QueryText = Annotated[
    str,
    typer.Argument(
        help="Natural language question to ask against the compiled wiki.",
    ),
]

QueryTopK = Annotated[
    int,
    typer.Option(
        "--top-k",
        help="Number of context chunks to retrieve (default: 5).",
        min=1,
        max=50,
    ),
]

QueryRaw = Annotated[
    bool,
    typer.Option(
        "--raw",
        help="Return raw context chunks without LLM synthesis.",
    ),
]

QueryFormatOption = Annotated[
    LlmDataFormat | None,
    typer.Option(
        "--format",
        help="Output format for agents: markdown (default), json, toon, ison. Defaults to query.format from config.",
    ),
]

QueryNoClassify = Annotated[
    bool,
    typer.Option(
        "--no-classify",
        help="Skip intent classification (always retrieve).",
    ),
]


class SearchMode(StrEnum):
    WIKI = "wiki"
    RAW = "raw"
    HYBRID = "hybrid"
    GRAPH = "graph"


QuerySearchMode = Annotated[
    SearchMode,
    typer.Option(
        "--search-mode",
        "--scope",
        help="Search mode: wiki (compiled pages), raw (source files), hybrid (both), or graph (hybrid + graph traversal expansion).",
    ),
]

QueryAdaptive = Annotated[
    bool,
    typer.Option(
        "--adaptive",
        help="Use adaptive retrieval (context-budget-aware zoom-in/zoom-out/targeted).",
    ),
]

QueryRemember = Annotated[
    bool,
    typer.Option(
        "--remember",
        help="Save this query and synthesized answer to the memory store.",
    ),
]

QueryVerbosity = Annotated[
    Verbosity | None,
    typer.Option(
        "--verbosity",
        help="Answer verbosity: brief, standard, or comprehensive. Defaults to search.synthesis.verbosity from config.",
    ),
]
