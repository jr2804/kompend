"""CLI argument types for the ``config`` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

ConfigKeyArg = Annotated[
    str,
    typer.Argument(
        help="Dotted config key, e.g. compile.budget.chunk_size_min",
    ),
]


ConfigValueArg = Annotated[
    str,
    typer.Argument(
        help="Value to set",
    ),
]


ConfigGlobalOpt = Annotated[
    bool,
    typer.Option(
        "--global",
        help="Operate on the global config instead of the project config.",
    ),
]

ConfigCheckPaths = Annotated[
    list[Path] | None,
    typer.Argument(
        help=("TOML files to diff. Omit to check the discovered project config. Multiple files merge (later wins); use --no-merge for per-file reports."),
    ),
]

ConfigNoMerge = Annotated[
    bool,
    typer.Option(
        "--no-merge",
        help="Report on each file independently instead of merging them.",
    ),
]
