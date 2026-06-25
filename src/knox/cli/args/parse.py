"""CLI arguments for the parse command."""

from __future__ import annotations

from typing import Annotated

import typer

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
