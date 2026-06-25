"""CLI arguments for the compile command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from knox.core.pipeline_defs import ImageMode as ImageModeEnum
from knox.core.pipeline_defs import OutputFormat as OutputFormatEnum

# ── `compile` command surface ───────────────────────────────────────────

CompileFiles = Annotated[
    list[str],
    typer.Argument(
        help="Files, glob patterns, or directories to compile. If only directories are given, all supported files inside are used.",
    ),
]

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

TargetFormat = Annotated[
    OutputFormatEnum,
    typer.Option(
        "--format",
        "-f",
        help="Output format: markdown, json, typst.",
    ),
]

ImageMode = Annotated[
    ImageModeEnum,
    typer.Option(
        "--image-mode",
        help="Image handling: file, base64.",
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
    bool,
    typer.Option(
        "--index",
        help="Index compiled wiki in zvec for semantic search.",
    ),
]

EmbedModel = Annotated[
    str | None,
    typer.Option(
        "--embed-model",
        help="Embedding model for zvec indexing.",
    ),
]

EmbedBatchSize = Annotated[
    int | None,
    typer.Option(
        "--embed-batch-size",
        help="Batch size for embedding generation during indexing.",
        min=1,
        max=1000,
        rich_help_panel="Advanced",
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
    bool,
    typer.Option(
        "--synthesize",
        help="Also generate an LLM-synthesized overview.md page.",
    ),
]

# ── N13: multi-flavor Markdown via rumdl ────────────────────────────────

OutputFlavor = Annotated[
    str | None,
    typer.Option(
        "--flavor",
        help=(
            "Markdown output flavor: obsidian, gfm, mkdocs, mdx, quarto, myst, "
            "standard, pandoc, kramdown, azure_devops. "
            "Auto-detected from output dir when not set."
        ),
    ),
]

# ── N1 compile --lint integration ───────────────────────────────────────

CompileLint = Annotated[
    bool,
    typer.Option(
        "--lint",
        help="Run lint after compilation and report quality issues.",
    ),
]
