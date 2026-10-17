"""CLI arguments for the knowledge commands (attachments, references, capture,
pack, organize, agents).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

#: Default for ``write --source``: an empty list, named so the alias default
#: is not a bare mutable literal in a signature (B006).
WRITE_SOURCES_DEFAULT: list[str] = []

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

# ── `attachments` command surface ───────────────────────────────────────

AttachmentSource = Annotated[
    str,
    typer.Argument(help="Source file name or relative path (e.g., 'report.pdf')."),
]

AttachmentFile = Annotated[
    Path,
    typer.Argument(help="Path to the attachment file to register."),
]

AttachmentIngestible = Annotated[
    bool,
    typer.Option("--ingest", help="Mark attachment as ingestible (processed during compile)."),
]

AttachmentEmbed = Annotated[
    bool | None,
    typer.Option("--embed/--no-embed", help="Override global embed_attachments setting for this attachment."),
]

AttachmentPath = Annotated[
    str,
    typer.Argument(help="Attachment relative path (e.g., 'report/photo.jpg') or just filename."),
]

AttachmentPathOptional = Annotated[
    str | None,
    typer.Argument(help="Attachment relative path (e.g., 'report/photo.jpg') or just filename. Omit with --all."),
]

AttachmentAll = Annotated[
    bool,
    typer.Option("--all", help="Apply to every registered attachment."),
]

AttachmentListSource = Annotated[
    str | None,
    typer.Option("--source", "-s", help="Filter by source name."),
]

AttachmentListAll = Annotated[
    bool,
    typer.Option("--all", "-a", help="Include deleted entries."),
]

# ── `references` command surface ────────────────────────────────────────

ReferencesListSource = Annotated[
    str | None,
    typer.Option("--source", "-s", help="Filter to references from this source file."),
]

ReferencesExportOutput = Annotated[
    Path,
    typer.Option("--output", "-o", help="Output file path. Default: references.bib"),
]

ReferencesExportSource = Annotated[
    str | None,
    typer.Option("--source", "-s", help="Export only references from this source."),
]

ReferencesFetchSource = Annotated[
    str | None,
    typer.Option("--source", "-s", help="Only resolve references from this source file."),
]

ReferencesFetchDryRun = Annotated[
    bool,
    typer.Option("--dry-run", "-n", help="Resolve but do not save results to disk."),
]

# ── `consolidate` / `distill` / `write` command surface ────────────────

ConsolidateDryRun = Annotated[
    bool,
    typer.Option("--dry-run", help="Report duplicate groups without moving anything."),
]

ConsolidateNoArchive = Annotated[
    bool,
    typer.Option("--no-archive", help="Keep duplicate notes in place (report only)."),
]

DistillNoArchive = Annotated[
    bool,
    typer.Option("--no-archive", help="Keep the source notes in place after distilling (default: archive them to sources/_archive/)."),
]

WritePageTitle = Annotated[
    str,
    typer.Argument(help="Title of the page to create."),
]

WriteSourceFiles = Annotated[
    list[str],
    typer.Option("--source", "-s", help="Source file(s) to include (repeatable)."),
]

# ── `pack-install` command surface ──────────────────────────────────────

PackName = Annotated[
    str,
    typer.Argument(help="Pack name to install (e.g. 'academic')."),
]
