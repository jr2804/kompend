"""CLI arguments for tiered compilation and pipeline control."""

from __future__ import annotations

from typing import Annotated

import typer

from knox.core.config.keys import EnvVar

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
        help="Bypass the parse cache and re-parse every source. Default: cache on.",
        rich_help_panel="Pipeline",
    ),
]

Workers = Annotated[
    int,
    typer.Option(
        "--workers",
        "-j",
        help="Number of parallel parse workers (1 = sequential).",
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

InPlaceFlag = Annotated[
    bool | None,
    typer.Option(
        "--in-place/--no-in-place",
        help="Keep source files at their original location instead of copying to sources/. Default: off (copy into sources/).",
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
