"""CLI arguments for the knowledge commands (attachments, references, capture,
pack, organize, agents).
"""

from __future__ import annotations

from typing import Annotated

import typer

# ── `organize` command surface ──────────────────────────────────────────

OrganizeMaxDepth = Annotated[
    int | None,
    typer.Option(
        "--max-depth",
        help="Maximum category tree depth (1-4). Env: KOMP_ORGANIZE_MAX_DEPTH.",
        min=1,
        max=4,
    ),
]

OrganizeMinSubcategories = Annotated[
    int | None,
    typer.Option(
        "--min-subcategories",
        help="Minimum sub-categories required to justify a new level. Env: KOMP_ORGANIZE_MIN_SUBCATEGORIES.",
        min=2,
    ),
]

OrganizeDryRun = Annotated[
    bool,
    typer.Option(
        "--dry-run",
        help="Show the proposed category tree without moving files or rewriting wiki links.",
    ),
]

OrganizeCollapseSingle = Annotated[
    bool | None,
    typer.Option(
        "--collapse-single-file/--no-collapse-single-file",
        help="Collapse single-file subcategories into the parent directory.",
    ),
]

# ── Q22 capture/learn/ingest commands ────────────────────────────────────

CaptureText = Annotated[
    str,
    typer.Argument(
        help="Text to extract knowledge items from.",
    ),
]

LearnText = Annotated[
    str,
    typer.Argument(
        help="Fact or snippet to store directly as a source file.",
    ),
]

SourceTags = Annotated[
    list[str] | None,
    typer.Option(
        "--tags",
        help=(
            "Source tags (repeatable; comma-separated within one option is AND, "
            "multiple --tags flags are OR). Blanks stripped from tag names; quote "
            "tags containing spaces to preserve them."
        ),
    ),
]

SourceAttrs = Annotated[
    list[str] | None,
    typer.Option(
        "--attr",
        help=("Source attributes as KEY=VALUE (repeatable). Multiple --attr flags are AND — each sets one key=value pair on the source."),
    ),
]

IngestTarget = Annotated[
    str,
    typer.Argument(
        help="URL or file path to ingest into the sources directory.",
    ),
]
