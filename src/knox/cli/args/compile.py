"""CLI arguments for the compile command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

# ── `compile` command surface ───────────────────────────────────────────

OnlineUrls = Annotated[
    list[str] | None,
    typer.Option(
        "--url",
        help="URLs to download before processing (repeatable).",
        rich_help_panel="Sources",
    ),
]

UrlFile = Annotated[
    Path | None,
    typer.Option(
        "--url-file",
        help="File with URLs to download (one per line, # comments ignored).",
        rich_help_panel="Sources",
    ),
]

Ignore = Annotated[
    list[str] | None,
    typer.Option(
        "--ignore",
        help="Ignore pattern (glob-style, repeatable).",
        rich_help_panel="Sources",
    ),
]

IndexStore = Annotated[
    bool | None,
    typer.Option(
        "--index/--no-index",
        help="Index compiled wiki in zvec for semantic search. Default: on.",
        rich_help_panel="Output",
    ),
]

EmbedModel = Annotated[
    str | None,
    typer.Option(
        "--embed-model",
        help="Embedding model for zvec indexing. Default: config compile.embedding.model (empty = no vector index).",
        rich_help_panel="Output",
    ),
]

# ── Q18: custom frontmatter fields ──────────────────────────────────────

Frontmatter = Annotated[
    list[str] | None,
    typer.Option(
        "--frontmatter",
        help=("Frontmatter field in key=value format (repeatable). E.g. --frontmatter language=en --frontmatter domain=ml"),
    ),
]

SynthesizeFlag = Annotated[
    bool | None,
    typer.Option(
        "--synthesize/--no-synthesize",
        help="Also generate an LLM-synthesized overview.md page. Default: on.",
        rich_help_panel="Output",
    ),
]

# ── N1 compile --lint integration ───────────────────────────────────────

CompileLint = Annotated[
    bool,
    typer.Option(
        "--lint/--no-lint",
        help="Run deterministic structural lint (no LLM) after compile. Default: on (override with KNOX_COMPILE_LINT).",
        rich_help_panel="Output",
        envvar="KNOX_COMPILE_LINT",
    ),
]

CompileLintAll = Annotated[
    bool,
    typer.Option(
        "--lint-all/--no-lint-all",
        help="Run the full lint pipeline (structural + LLM rules + remediate) after compile. Implies --lint. Override with KNOX_COMPILE_LINT_ALL.",
        rich_help_panel="Output",
        envvar="KNOX_COMPILE_LINT_ALL",
    ),
]
