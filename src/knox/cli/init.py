"""``knox init`` — scaffold a new wiki project."""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

import typer

from knox.cli.app import app
from knox.cli.args_init import InitBare, InitForce, InitGit, InitTarget
from knox.scaffolding import ScaffoldError, scaffold_project

logger = logging.getLogger(__name__)


_NEXT_STEPS_RESOURCE: str = "next_steps.txt"


def _load_next_steps() -> str:
    return resources.files("knox.scaffolding.templates").joinpath(_NEXT_STEPS_RESOURCE).read_text(encoding="utf-8")


@app.command(name="init", help="Scaffold a new KNOX wiki project.")
def init(
    target: InitTarget = ".",
    bare: InitBare = False,
    force: InitForce = False,
    git: InitGit = False,
) -> None:
    """Create a new wiki project in TARGET (default: current directory)."""
    try:
        result = scaffold_project(Path(target), bare=bare, force=force, git=git)
    except ScaffoldError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Created wiki project at {result.target_dir}")
    for directory in result.created_dirs:
        typer.echo(f"  + {directory.relative_to(result.target_dir)}/")
    for file in result.copied_files:
        typer.echo(f"  + {file.relative_to(result.target_dir)}")
    if result.skipped_files:
        typer.echo(f"  - skipped (--bare): {', '.join(result.skipped_files)}")
    if result.git_initialised:
        typer.echo("  + git init: initial commit created")
    typer.echo("")
    typer.echo(_load_next_steps(), nl=False)
