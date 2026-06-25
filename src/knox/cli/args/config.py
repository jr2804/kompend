"""CLI argument types for the ``config`` subcommand."""

from __future__ import annotations

from typing import Annotated

import typer

ConfigKeyArg = Annotated[
    str,
    typer.Argument(
        help="Dotted config key, e.g. compile.chunk_size",
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
