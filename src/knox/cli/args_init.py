"""CLI argument types for the ``init`` subcommand."""

from __future__ import annotations

from typing import Annotated

import typer

InitTarget = Annotated[
    str,
    typer.Argument(
        help="Project directory. Created if it does not exist. Defaults to CWD.",
    ),
]

InitBare = Annotated[
    bool,
    typer.Option(
        "--bare",
        help="Skip authoring templates (NOTES.md, README.md).",
    ),
]

InitForce = Annotated[
    bool,
    typer.Option(
        "--force",
        "-f",
        help="Overwrite an existing project marker (knox.toml or .knox/config.toml).",
    ),
]

InitGit = Annotated[
    bool,
    typer.Option(
        "--git",
        help="Run 'git init' and create an initial commit after scaffolding.",
    ),
]
