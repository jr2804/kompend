"""CLI arguments for the knowledge commands (attachments, references, capture,
pack, organize, agents).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from kompend.cli.args.common import in_range

#: Default for ``write --source``: an empty list, named so the alias default
#: is not a bare mutable literal in a signature (B006).
WRITE_SOURCES_DEFAULT: list[str] = []

# ── `organize` command surface ──────────────────────────────────────────

OrganizeMaxDepth = Annotated[
    int | None,
    Parameter(
        name="--max-depth",
        help="Maximum category tree depth (1-4). Env: KOMP_ORGANIZE_MAX_DEPTH.",
        validator=in_range(1, 4),
    ),
]

OrganizeMinSubcategories = Annotated[
    int | None,
    Parameter(
        name="--min-subcategories",
        help="Minimum sub-categories required to justify a new level. Env: KOMP_ORGANIZE_MIN_SUBCATEGORIES.",
        validator=in_range(2),
    ),
]

OrganizeDryRun = Annotated[
    bool,
    Parameter(
        name="--dry-run",
        help="Show the proposed category tree without moving files or rewriting wiki links.",
        negative="",
    ),
]

OrganizeCollapseSingle = Annotated[
    bool | None,
    Parameter(
        name="--collapse-single-file",
        help="Collapse single-file subcategories into the parent directory.",
    ),
]

# ── Q22 capture/learn/ingest commands ────────────────────────────────────

CaptureText = Annotated[
    str,
    Parameter(
        help="Text to extract knowledge items from.",
    ),
]

LearnText = Annotated[
    str,
    Parameter(
        help="Fact or snippet to store directly as a source file.",
    ),
]

SourceTags = Annotated[
    list[str] | None,
    Parameter(
        negative="",
        name="--tags",
        help=(
            "Source tags (repeatable; comma-separated within one option is AND, "
            "multiple --tags flags are OR). Blanks stripped from tag names; quote "
            "tags containing spaces to preserve them."
        ),
    ),
]

SourceAttrs = Annotated[
    list[str] | None,
    Parameter(
        negative="",
        name="--attr",
        help="Source attributes as KEY=VALUE (repeatable). Multiple --attr flags are AND — each sets one key=value pair on the source.",
    ),
]

IngestTarget = Annotated[
    str,
    Parameter(
        help="URL or file path to ingest into the sources directory.",
    ),
]

# ── `attachments` command surface ───────────────────────────────────────

AttachmentSource = Annotated[
    str,
    Parameter(help="Source file name or relative path (e.g., 'report.pdf')."),
]

AttachmentFile = Annotated[
    Path,
    Parameter(help="Path to the attachment file to register."),
]

AttachmentIngestible = Annotated[
    bool,
    Parameter(name="--ingest", help="Mark attachment as ingestible (processed during compile).", negative=""),
]

AttachmentEmbed = Annotated[
    bool | None,
    Parameter(name="--embed", help="Override global embed_attachments setting for this attachment."),
]

AttachmentPath = Annotated[
    str,
    Parameter(help="Attachment relative path (e.g., 'report/photo.jpg') or just filename."),
]

AttachmentPathOptional = Annotated[
    str | None,
    Parameter(help="Attachment relative path (e.g., 'report/photo.jpg') or just filename. Omit with --all."),
]

AttachmentAll = Annotated[
    bool,
    Parameter(name="--all", help="Apply to every registered attachment.", negative=""),
]

AttachmentListSource = Annotated[
    str | None,
    Parameter(name=("--source", "-s"), help="Filter by source name."),
]

AttachmentListAll = Annotated[
    bool,
    Parameter(name=("--all", "-a"), help="Include deleted entries.", negative=""),
]

# ── `references` command surface ────────────────────────────────────────

ReferencesListSource = Annotated[
    str | None,
    Parameter(name=("--source", "-s"), help="Filter to references from this source file."),
]

ReferencesExportOutput = Annotated[
    Path,
    Parameter(name=("--output", "-o"), help="Output file path. Default: references.bib"),
]

ReferencesExportSource = Annotated[
    str | None,
    Parameter(name=("--source", "-s"), help="Export only references from this source."),
]

ReferencesFetchSource = Annotated[
    str | None,
    Parameter(name=("--source", "-s"), help="Only resolve references from this source file."),
]

ReferencesFetchDryRun = Annotated[
    bool,
    Parameter(name=("--dry-run", "-n"), help="Resolve but do not save results to disk.", negative=""),
]

# ── `consolidate` / `distill` / `write` command surface ────────────────

ConsolidateDryRun = Annotated[
    bool,
    Parameter(name="--dry-run", help="Report duplicate groups without moving anything.", negative=""),
]

ConsolidateNoArchive = Annotated[
    bool,
    Parameter(name="--no-archive", help="Keep duplicate notes in place (report only).", negative=""),
]

DistillNoArchive = Annotated[
    bool,
    Parameter(
        name="--no-archive",
        help="Keep the source notes in place after distilling (default: archive them to sources/_archive/).",
        negative="",
    ),
]

WritePageTitle = Annotated[
    str,
    Parameter(help="Title of the page to create."),
]

WriteSourceFiles = Annotated[
    list[str],
    Parameter(negative="", name=("--source", "-s"), help="Source file(s) to include (repeatable)."),
]

# ── `pack-install` command surface ──────────────────────────────────────

PackName = Annotated[
    str,
    Parameter(help="Pack name to install (e.g. 'academic')."),
]
