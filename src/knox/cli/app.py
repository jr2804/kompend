"""Typer CLI app — lazy command registration for fast startup.

Command modules are imported on-demand: only the module for the invoked
subcommand is loaded, so ``knox --version`` or ``knox config get`` avoid
loading heavy deps (pydantic-ai, docling, MCP SDK) when not needed.
``--help`` and bare ``knox`` (no args) still load everything
because the full command tree must be visible.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from importlib.metadata import version

import typer

from knox.core.reporting.runlog import log_run_end, log_run_start
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
    _register_for_argv(sys.argv[1:])
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


# ── Lazy command registry ─────────────────────────────────────────────
#
# Maps subcommand name → module path.  Importing the module registers its
# command(s) on ``app`` via the ``@app.command()`` decorator side effect.
# Multiple subcommands can share one module (e.g. ``compile`` + ``ingest`` +
# ``parse`` all live in ``pipeline.py``).
#
# Modules NOT listed here are never imported by the CLI — add new commands
# here when they are created.

_LAZY_MODULES: dict[str, str] = {
    "agents": "knox.cli.cmds.agents",
    "capture": "knox.cli.cmds.knowledge.capture",
    "learn": "knox.cli.cmds.knowledge.capture",
    "distill": "knox.cli.cmds.knowledge.capture",
    "write": "knox.cli.cmds.knowledge.capture",
    "config": "knox.cli.cmds.pipeline.config",
    "doctor": "knox.cli.cmds.quality.doctor_cmd",
    "status": "knox.cli.cmds.quality.status_cmd",
    "list-parsers": "knox.cli.cmds.quality.list_parsers_cmd",
    "trace": "knox.cli.cmds.quality.trace_cmd",
    "init": "knox.cli.cmds.init",
    "lint": "knox.cli.cmds.quality.lint",
    "log": "knox.cli.cmds.quality.log",
    "memory": "knox.cli.cmds.knowledge.memory",
    "organize": "knox.cli.cmds.knowledge.organize",
    "pack-list": "knox.cli.cmds.knowledge.pack",
    "pack-install": "knox.cli.cmds.knowledge.pack",
    "parse": "knox.cli.cmds.pipeline.pipeline",
    "compile": "knox.cli.cmds.pipeline.pipeline",
    "ingest": "knox.cli.cmds.pipeline.pipeline",
    "query": "knox.cli.cmds.knowledge.query",
    "review": "knox.cli.cmds.quality.review",
    "sources": "knox.cli.cmds.pipeline.sources",
    "sources-tag": "knox.cli.cmds.pipeline.sources",
    "sources-untag": "knox.cli.cmds.pipeline.sources",
    "mcp": "knox.cli.cmds.serve",
    "serve": "knox.cli.cmds.serve",
    "agent": "knox.cli.cmds.agent",
}


def register_command(name: str) -> None:
    """Import the module that registers *name*.  No-op if unknown or already loaded."""
    module_path = _LAZY_MODULES.get(name)
    if module_path is not None:
        importlib.import_module(module_path)


def register_all_commands() -> None:
    """Import every command module (for ``--help``, bare ``knox``, shell completion)."""
    for module_path in dict.fromkeys(_LAZY_MODULES.values()):
        importlib.import_module(module_path)


def _find_requested_command(argv: list[str]) -> str | None:
    """Return the subcommand name from *argv*, or *None* if none is present.

    Handles global flags before the subcommand (``knox --wiki-dir x compile``).
    """
    for arg in argv:
        if arg in _LAZY_MODULES:
            return arg
    return None


def _register_for_argv(argv: list[str]) -> None:
    """Lazy-register commands based on the incoming argv.

    * ``--help`` / ``-h`` / empty argv → load all (help must list every command).
    * Shell-completion env vars (``*_COMPLETE``) → load all.
    * A known subcommand → load only its module.
    * Otherwise (global flags only, e.g. ``--version``) → load nothing.
    """
    if not argv or argv[0] in ("--help", "-h") or any(k.endswith("_COMPLETE") for k in os.environ):
        register_all_commands()
        return
    requested = _find_requested_command(argv)
    if requested is not None:
        register_command(requested)
