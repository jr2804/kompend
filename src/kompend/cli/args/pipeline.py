"""CLI arguments for tiered compilation and pipeline control."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from kompend.cli.args.common import PIPELINE_GROUP, SOURCES_GROUP, in_range
from kompend.core.keys import EnvVar
from kompend.core.vocab import ExportMode

# ── Q20 tiered compilation ──────────────────────────────────────────────

StartAtTier = Annotated[
    str,
    Parameter(
        name="--start-at-tier",
        help=(
            "Pipeline entry point: index | chunk | resolve | compile | output. "
            "Tiers before this are skipped (cached results reused). "
            "Previous tiers must have been completed in a prior run."
        ),
        env_var=EnvVar.START_AT_TIER,
        group=PIPELINE_GROUP,
    ),
]

ReparseFlag = Annotated[
    bool | None,
    Parameter(
        name="--reparse",
        help=(
            "Re-parse every source instead of reusing the cached parse. "
            "Sources are re-extracted and re-chunked; cached LLM results are kept. "
            "Tags/attributes are preserved; attachments are untouched."
        ),
        group=PIPELINE_GROUP,
    ),
]

Workers = Annotated[
    int,
    Parameter(
        name=("--workers", "-j"),
        help="Number of parallel parse workers (1 = sequential). Threads only run Python code in parallel on a free-threaded interpreter (CPython 3.14t+).",
        validator=in_range(1),
        group=PIPELINE_GROUP,
    ),
]

ForceTier = Annotated[
    bool | None,
    Parameter(
        name="--force",
        help="Re-process every source regardless of persisted tier state. Default: off (reuse cached results).",
        group=PIPELINE_GROUP,
    ),
]

IgnoreModelFlag = Annotated[
    bool | None,
    Parameter(
        name="--ignore-model",
        help=(
            "Reuse cached extractions, summaries and generated pages across model switches "
            "(e.g. finish a run on a local model after an online quota ran out). Cached "
            "artifacts are keyed without the model; pages keep the model tag they were "
            "generated with. Default: off (a model switch invalidates cached LLM products)."
        ),
        group=PIPELINE_GROUP,
    ),
]

InPlaceFlag = Annotated[
    bool | None,
    Parameter(
        name="--in-place",
        help="Keep source files at their original location instead of copying to sources/. Default: off (copy into sources/).",
        group=SOURCES_GROUP,
    ),
]

ListTiers = Annotated[
    bool,
    Parameter(
        name="--list-tiers",
        help="Show the available pipeline tiers and their current state, then exit.",
        group=PIPELINE_GROUP,
        negative="",
    ),
]

JsonProgress = Annotated[
    bool,
    Parameter(
        name="--json-progress",
        help="Emit progress as JSONL to stdout (for machine consumers).",
        group=PIPELINE_GROUP,
        negative="",
    ),
]

# ── sources export ───────────────────────────────────────────────────────

ExportAll = Annotated[
    bool,
    Parameter(
        name="--all",
        help="Export every ingested source into exported/.",
        negative="",
    ),
]

ExportInline = Annotated[
    bool,
    Parameter(
        name="--inline",
        help="Embed images as data URIs (default: copy to <output stem>_assets/).",
    ),
]

ExportAnnotations = Annotated[
    ExportMode | None,
    Parameter(
        name="--annotations",
        help="Where extracted table/figure annotations go: off, inline, or asset (default: export.annotations config).",
    ),
]

ExportSummary = Annotated[
    ExportMode | None,
    Parameter(
        name="--summary",
        help="Where the document summary goes: off, inline, or asset (default: export.summary config).",
    ),
]

# ── ingest / sources commands ──────────────────────────────────────────

IngestFiles = Annotated[
    list[str] | None,
    Parameter(
        negative="",
        help="Files, glob patterns, or directories to ingest. If omitted, all supported files in the wiki sources dir are scanned.",
    ),
]

ExportSourceId = Annotated[
    str,
    Parameter(),
]


MemoryEnabled = Annotated[
    bool | None,
    Parameter(
        name="--memory",
        help="Track node lifecycle (confidence, tiers, supersession, retention/decay) across compiles. Disabled skips the memory update pass. Default: on.",
        env_var=EnvVar.MEMORY_ENABLED,
        group=PIPELINE_GROUP,
    ),
]
