"""CLI argument types for the ``init`` subcommand."""

from __future__ import annotations

from pathlib import Path
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
        help="Overwrite an existing project marker (kompend.toml or .kompend/config.toml).",
    ),
]

InitKeepCache = Annotated[
    bool,
    typer.Option(
        "--keep-cache",
        help=(
            "With --force: remove all contents of the target directory "
            "before scaffolding, then restore the previous .kompend/cache/. "
            "Enables fast re-runs where ingest skips already-parsed documents."
        ),
    ),
]

InitGit = Annotated[
    bool,
    typer.Option(
        "--git",
        help="Run 'git init' and create an initial commit after scaffolding.",
    ),
]

InitConfig = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help=(
            "Path to a TOML config to install as the vault's .kompend/config.toml "
            "instead of the generated default. Use to seed a vault with a chosen "
            "LLM/budget/profile (e.g. scripts/configs/config-cloud.toml). The file is "
            "copied verbatim."
        ),
    ),
]

InitWikiDir = Annotated[
    str | None,
    typer.Option(
        "--wiki-dir",
        "-w",
        help=(
            "Project directory to scaffold into, as an alternative to the TARGET "
            "argument. Mirrors the global --wiki-dir/-w option, which must precede "
            "the subcommand. Passing both with different paths is an error."
        ),
    ),
]
