"""CLI arguments for tiered compilation and pipeline control."""

from __future__ import annotations

from typing import Annotated

import typer

from kompend.core.keys import EnvVar
from kompend.core.vocab import ExportMode

# ── Q20 tiered compilation ──────────────────────────────────────────────

StartAtTier = Annotated[
    str,
    typer.Option(
        "--start-at-tier",
        help=(
            "Pipeline entry point: index | chunk | resolve | compile | output. "
            "Tiers before this are skipped (cached results reused). "
            "Previous tiers must have been completed in a prior run."
        ),
        envvar=EnvVar.START_AT_TIER,
        rich_help_panel="Pipeline",
    ),
]

NoCacheFlag = Annotated[
    bool | None,
    typer.Option(
        "--no-cache/--cache",
        help=("Force a re-parse of every source (bypass the parse cache). Tags/attributes are preserved; attachments are untouched."),
        rich_help_panel="Pipeline",
    ),
]

Workers = Annotated[
    int,
    typer.Option(
        "--workers",
        "-j",
        help=("Number of parallel parse workers (1 = sequential). Threads only run Python code in parallel on a free-threaded interpreter (CPython 3.14t+)."),
        min=1,
        rich_help_panel="Pipeline",
    ),
]

ForceTier = Annotated[
    bool | None,
    typer.Option(
        "--force/--no-force",
        help="Re-process every source regardless of persisted tier state. Default: off (reuse cached results).",
        rich_help_panel="Pipeline",
    ),
]

IgnoreModelFlag = Annotated[
    bool | None,
    typer.Option(
        "--ignore-model/--no-ignore-model",
        help=(
            "Reuse cached extractions, summaries and generated pages across model switches "
            "(e.g. finish a run on a local model after an online quota ran out). Cached "
            "artifacts are keyed without the model; pages keep the model tag they were "
            "generated with. Default: off (a model switch invalidates cached LLM products)."
        ),
        rich_help_panel="Pipeline",
    ),
]

InPlaceFlag = Annotated[
    bool | None,
    typer.Option(
        "--in-place/--no-in-place",
        help="Keep source files at their original location instead of copying to sources/. Default: off (copy into sources/).",
        rich_help_panel="Sources",
    ),
]

RecursiveScan = Annotated[
    bool | None,
    typer.Option(
        "--recursive/--no-recursive",
        help="Scan sources/ subdirectories for source files. Default: paths.recursive config value (on).",
        rich_help_panel="Sources",
    ),
]

ListTiers = Annotated[
    bool,
    typer.Option(
        "--list-tiers",
        help="Show the available pipeline tiers and their current state, then exit.",
        rich_help_panel="Pipeline",
    ),
]

JsonProgress = Annotated[
    bool,
    typer.Option(
        "--json-progress",
        help="Emit progress as JSONL to stdout (for machine consumers).",
        rich_help_panel="Pipeline",
    ),
]

# ── sources export ───────────────────────────────────────────────────────

ExportAll = Annotated[
    bool,
    typer.Option("--all", help="Export every ingested source into exported/."),
]

ExportInline = Annotated[
    bool,
    typer.Option(
        "--inline/--no-inline",
        help="Embed images as data URIs (default: copy to <output stem>_assets/).",
    ),
]

ExportAnnotations = Annotated[
    ExportMode | None,
    typer.Option(
        "--annotations",
        help="Where extracted table/figure annotations go: off, inline, or asset (default: export.annotations config).",
    ),
]

ExportSummary = Annotated[
    ExportMode | None,
    typer.Option(
        "--summary",
        help="Where the document summary goes: off, inline, or asset (default: export.summary config).",
    ),
]

# ── ingest / sources commands ──────────────────────────────────────────

IngestFiles = Annotated[
    list[str] | None,
    typer.Argument(
        help=("Files, glob patterns, or directories to ingest. If omitted, all supported files in the wiki sources dir are scanned."),
    ),
]

ExportSourceId = Annotated[
    str,
    typer.Argument(),
]


MemoryEnabled = Annotated[
    bool | None,
    typer.Option(
        "--memory/--no-memory",
        help=("Track node lifecycle (confidence, tiers, supersession, retention/decay) across compiles. Disabled skips the memory update pass. Default: on."),
        envvar=EnvVar.MEMORY_ENABLED,
        rich_help_panel="Pipeline",
    ),
]
