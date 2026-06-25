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
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
)


@app.callback(invoke_without_command=True)
def _callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
        is_eager=True,
    ),
    wiki_dir: str | None = typer.Option(
        None,
        "--wiki-dir",
        "-w",
        help="Project root directory. All other paths are resolved relative to it. Default: current working directory. Env: KNOX_WIKI_DIR.",
        envvar="KNOX_WIKI_DIR",
    ),
    no_user_tracking: bool = typer.Option(
        False,
        "--no-user-tracking",
        help="Do not record the OS user (ingested_by / memory_by) in metadata for this invocation. Env: KNOX_USER_TRACKING=false.",
    ),
) -> None:
    """Knowledge Organizer for eXtended documents"""
    if version:
        typer.echo(_get_version())
        raise typer.Exit()
    ctx.obj = {"wiki_dir": wiki_dir, "no_user_tracking": no_user_tracking}


def main() -> None:
    """Entry point for the CLI application."""
    app()


from knox.cli.cmds import (  # noqa: E402
    capture,  # noqa: F401
    config,  # noqa: F401
    doctor,  # noqa: F401
    init,  # noqa: F401
    lint,  # noqa: F401
    memory,  # noqa: F401
    organize,  # noqa: F401
    pack,  # noqa: F401
    pipeline,  # noqa: F401
    quality,  # noqa: F401
    query,  # noqa: F401
    review,  # noqa: F401
    serve,  # noqa: F401
    status,  # noqa: F401
)
