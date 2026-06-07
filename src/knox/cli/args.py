"""CLI arguments, options, and defaults for KNOX.

All Typer ``Argument`` and ``Option`` declarations live here as
``Annotated`` types. Command files import the types and apply them as
plain annotations — this keeps B008 happy and centralises the surface
area that users see in ``--help``.
"""

from __future__ import annotations

from typing import Annotated

import typer

# Application name used for environment variable prefixes.
APP_NAME_UPPERCASE = "KNOX"

# ── Defaults ────────────────────────────────────────────────────────────

DEFAULT_SOURCES_DIR: str = "sources"
DEFAULT_OUTPUT_DIR: str = "wiki"
DEFAULT_COMPILE_FORMAT: str = "markdown"
DEFAULT_CHUNK_SIZE: int = 512
DEFAULT_IMAGE_MODE: str = "file"


# ── Shared options (used by multiple commands) ──────────────────────────

Recursive = Annotated[
    bool,
    typer.Option(
        "--recursive/--no-recursive",
        "-r/-R",
        help="Search source directories recursively.",
        envvar=f"{APP_NAME_UPPERCASE}_RECURSIVE",
    ),
]

SourcesDir = Annotated[
    str | None,
    typer.Option(
        "--sources-dir",
        "-s",
        help="Input sources directory. Files not already here are copied in. "
        f"Default: '{DEFAULT_SOURCES_DIR}' in the current working directory. "
        f"Env: {APP_NAME_UPPERCASE}_SOURCES_DIR.",
    ),
]

OutputDir = Annotated[
    str | None,
    typer.Option(
        "--output-dir",
        "-o",
        help="Output directory for compiled wiki. "
        f"Default: '{DEFAULT_OUTPUT_DIR}' in the current working directory. "
        f"Env: {APP_NAME_UPPERCASE}_OUTPUT_DIR.",
    ),
]


# ── `parse` command surface ─────────────────────────────────────────────

ParseFiles = Annotated[
    list[str],
    typer.Argument(
        help="Files, glob patterns, or directories to parse.",
    ),
]

PrintElements = Annotated[
    bool,
    typer.Option(
        "--elements",
        "-e",
        help="Print extracted elements.",
    ),
]


# ── `compile` command surface ───────────────────────────────────────────

CompileFiles = Annotated[
    list[str],
    typer.Argument(
        help="Files, glob patterns, or directories to compile. "
        "If only directories are given, all supported files inside are used.",
    ),
]

OnlineUrls = Annotated[
    list[str] | None,
    typer.Option(
        "--url",
        help="URLs to download before processing (repeatable).",
    ),
]

TargetFormat = Annotated[
    str,
    typer.Option(
        "--format",
        "-f",
        help="Output format: markdown, json, typst.",
    ),
]

ChunkSize = Annotated[
    int,
    typer.Option(
        "--chunk-size",
        help="Chunk size in tokens.",
        min=1,
    ),
]

ImageMode = Annotated[
    str,
    typer.Option(
        "--image-mode",
        help="Image handling: file, base64.",
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
