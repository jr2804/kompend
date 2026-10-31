"""CLI argument types for the ``config`` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

ConfigKeyArg = Annotated[
    str,
    Parameter(
        help="Dotted config key, e.g. compile.budget.chunk_size_min",
    ),
]


ConfigValueArg = Annotated[
    str,
    Parameter(
        help="Value to set",
    ),
]


ConfigGlobalOpt = Annotated[
    bool,
    Parameter(
        name="--global",
        help="Operate on the global config instead of the project config.",
        negative="",
    ),
]

ConfigCheckPaths = Annotated[
    list[Path] | None,
    Parameter(
        negative="",
        help="TOML files to diff. Omit to check the discovered project config. Multiple files merge (later wins); use --no-merge for per-file reports.",
    ),
]

ConfigNoMerge = Annotated[
    bool,
    Parameter(
        name="--no-merge",
        help="Report on each file independently instead of merging them.",
        negative="",
    ),
]
