"""CLI arguments for the parse command."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

# ── `parse` command surface ─────────────────────────────────────────────

ParseFiles = Annotated[
    list[str],
    Parameter(
        negative="",
        help="Files, glob patterns, or directories to parse.",
    ),
]

PrintElements = Annotated[
    bool,
    Parameter(
        name=("--elements", "-e"),
        help="Print extracted elements.",
        negative="",
    ),
]
