"""CLI arguments for tiered compilation and pipeline control."""

from __future__ import annotations

from typing import Annotated

import typer

from .common import APP_NAME_UPPERCASE

# ── Q20 tiered compilation ──────────────────────────────────────────────

StartAtTier = Annotated[
    str,
    typer.Option(
        "--start-at-tier",
        help=(
            "Pipeline entry point: index | chunk | resolve | compile | output. "
            "Tiers before this are skipped (cached results reused). "
            "Previous tiers must have been completed in a prior run. "
            f"Env: {APP_NAME_UPPERCASE}_START_AT_TIER."
        ),
        envvar=f"{APP_NAME_UPPERCASE}_START_AT_TIER",
    ),
]

NoCacheFlag = Annotated[
    bool | None,
    typer.Option(
        "--no-cache/--cache",
        help="Bypass the parse cache and re-parse every source. Default: cache on.",
    ),
]

Workers = Annotated[
    int,
    typer.Option(
        "--workers",
        "-j",
        help="Number of parallel parse workers (1 = sequential).",
        min=1,
    ),
]

ForceTier = Annotated[
    bool | None,
    typer.Option(
        "--force/--no-force",
        help="Re-process every source regardless of persisted tier state.",
    ),
]

InPlaceFlag = Annotated[
    bool | None,
    typer.Option(
        "--in-place/--no-in-place",
        help="Keep source files at their original location instead of copying to sources/.",
    ),
]

ListTiers = Annotated[
    bool,
    typer.Option(
        "--list-tiers",
        help="Show the available pipeline tiers and their current state, then exit.",
    ),
]
