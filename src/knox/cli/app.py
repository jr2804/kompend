"""Typer CLI app for My Project."""

from __future__ import annotations

import contextlib
import sys
from importlib.metadata import version

import typer

from knox.core.runlog import log_run_end, log_run_start
from knox.logging import configure_logging, get_console


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
        get_console().print(_get_version())
        raise typer.Exit()
    ctx.obj = {"wiki_dir": wiki_dir, "no_user_tracking": no_user_tracking}


def main() -> None:
    """Entry point for the CLI application."""
    configure_logging()
    _configure_streams_utf8()
    handle = log_run_start()
    try:
        app()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        log_run_end(handle, status="ok" if code == 0 else "error", exit_code=code)
        raise
    except BaseException as exc:  # noqa: BLE001 — capture any crash for the audit trail
        log_run_end(handle, status="error", error=repr(exc))
        raise
    log_run_end(handle, status="ok", exit_code=0)


def _configure_streams_utf8() -> None:
    """Reconfigure stdout/stderr to UTF-8 so Unicode output never crashes.

    On Windows the default console codec (cp1252) cannot encode glyphs like
    ``✓`` / ``✗`` used in lint/memory output, which crashed Rich's renderer
    mid-flush and aborted the command before results were saved. Using
    ``errors="replace"`` guarantees no crash even on legacy consoles.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


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
