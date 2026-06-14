"""Typer CLI app for My Project."""

from __future__ import annotations

from importlib.metadata import version

import typer


# Version management
def _get_version() -> str:
    """Get application version from package metadata."""
    try:
        return version("knox")
    except Exception:
        return "0.0.0"  # Fallback for development mode


app = typer.Typer(
    name="knox",
    help="Knowledge Organizer for eXtended documents",
    add_completion=True,
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def _callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
        is_eager=True,
    ),
) -> None:
    """Knowledge Organizer for eXtended documents"""
    if version:
        typer.echo(_get_version())
        raise typer.Exit()


def main() -> None:
    """Entry point for the CLI application."""
    app()


# Import commands to register them with app
from knox.cli import (  # noqa: E402
    commands,  # noqa: F401
    config_cmd,  # noqa: F401
    init,  # noqa: F401
    memory_cmd,  # noqa: F401
)
