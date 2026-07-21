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
    ),
]

IndexStore = Annotated[
    bool | None,
    typer.Option(
        "--index/--no-index",
        help="Index compiled wiki in zvec for semantic search. Default: on.",
    ),
]

EmbedModel = Annotated[
    str | None,
    typer.Option(
        "--embed-model",
        help="Embedding model for zvec indexing.",
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
    ),
]

# ── N1 compile --lint integration ───────────────────────────────────────

CompileLint = Annotated[
    bool | None,
    typer.Option(
        "--lint/--no-lint",
        help="Run lint after compilation and report quality issues.",
    ),
]
