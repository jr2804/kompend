"""CLI argument types for the ``init`` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

InitTarget = Annotated[
    str,
    Parameter(
        help="Project directory. Created if it does not exist. Defaults to CWD.",
    ),
]

InitBare = Annotated[
    bool,
    Parameter(
        name="--bare",
        help="Skip authoring templates (NOTES.md, README.md).",
        negative="",
    ),
]

InitForce = Annotated[
    bool,
    Parameter(
        name=("--force", "-f"),
        help="Overwrite an existing project marker (kompend.toml or .kompend/config.toml).",
        negative="",
    ),
]

InitKeepCache = Annotated[
    bool,
    Parameter(
        name="--keep-cache",
        help=(
            "With --force: remove all contents of the target directory "
            "before scaffolding, then restore the previous .kompend/cache/. "
            "Enables fast re-runs where ingest skips already-parsed documents."
        ),
        negative="",
    ),
]

InitGit = Annotated[
    bool,
    Parameter(
        name="--git",
        help="Run 'git init' and create an initial commit after scaffolding.",
        negative="",
    ),
]

InitConfig = Annotated[
    Path | None,
    Parameter(
        name="--seed-config",
        help=(
            "Path to a TOML config to install as the vault's .kompend/config.toml "
            "instead of the generated default. Use to seed a vault with a chosen "
            "LLM/budget/profile (e.g. scripts/configs/demo.toml). The file is "
            "copied verbatim."
        ),
    ),
]

InitWikiDir = Annotated[
    str | None,
    Parameter(
        name=("--wiki-dir", "-w"),
        help=(
            "Project directory to scaffold into, as an alternative to the TARGET "
            "argument. Mirrors the global --wiki-dir/-w option, which must precede "
            "the subcommand. Passing both with different paths is an error."
        ),
    ),
]
