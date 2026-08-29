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

import logfire
import typer

from knox.core.config.keys import EnvVar
from knox.core.reporting.runlog import RunHandle, log_run_end, log_run_start
from knox.logging import configure_logging, get_console, get_err_console

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
    "attachments": "knox.cli.cmds.knowledge.attachments",
    "att": "knox.cli.cmds.knowledge.attachments",
    "capture": "knox.cli.cmds.knowledge.capture",
    "learn": "knox.cli.cmds.knowledge.capture",
    "distill": "knox.cli.cmds.knowledge.capture",
    "write": "knox.cli.cmds.knowledge.capture",
    "config": "knox.cli.cmds.pipeline.config",
    "doctor": "knox.cli.cmds.quality.doctor_cmd",
    "status": "knox.cli.cmds.quality.status_cmd",
    "list-parsers": "knox.cli.cmds.quality.list_parsers_cmd",
    "provenance": "knox.cli.cmds.quality.provenance_cmd",
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
    "index": "knox.cli.cmds.pipeline.pipeline",
    "query": "knox.cli.cmds.knowledge.query",
    "references": "knox.cli.cmds.knowledge.references",
    "review": "knox.cli.cmds.quality.review",
    "sources": "knox.cli.cmds.pipeline.sources",
    "mcp": "knox.cli.cmds.serve",
    "serve": "knox.cli.cmds.serve",
    "agent": "knox.cli.cmds.agent",
}


app = typer.Typer(
    name="knox",
    help="Knowledge Organizer for eXtended documents",
    add_completion=True,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
)


# Logfire instrumentation: configure() must run before instrument_pydantic_ai().
# We point at "knox" + "staging" so traces land in the starter project on
# logfire-eu. Respects KNOX_LOGFIRE_DISABLED and LOGFIRE_TOKEN.
def _configure_logfire() -> None:
    """Enable Logfire instrumentation once per process (called from main())."""
    if os.environ.get(EnvVar.LOGFIRE_DISABLED, "").strip().lower() in ("1", "true", "yes"):
        return
    # logfire fans telemetry out to standard OTLP exporters when the ambient
    # OTEL_EXPORTER_OTLP_ENDPOINT env var is set (logfire config.py:1469).
    # Each signal (traces/metrics/logs) has its own switch, all defaulting to
    # enabled. A stale collector endpoint from unrelated shell tooling
    # (e.g. Copilot's local collector, injected per-window into VS Code
    # terminals — never on disk) then receives knox telemetry and retries
    # against a dead port. knox has its own collector path
    # (agents/instrument.py, gated on KNOX_OTEL_ENABLED), so ambient OTel vars
    # must not divert logfire telemetry. setdefault keeps the escape hatch:
    # OTEL_TRACES_EXPORTER=otlp (etc.) explicitly re-enables fan-through.
    os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
    os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
    os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")
    logfire.configure(
        service_name="knox",
        environment="staging",
        send_to_logfire="if-token-present",
    )
    logfire.instrument_pydantic_ai()


#: Module-level holder for the run handle, written by :func:`_callback`
#: (where Typer has resolved ``--wiki-dir``) and read by :func:`main`.
_run_handle_holder: dict[str, RunHandle | None] = {"handle": None}


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
        help="Project root directory. All other paths are resolved relative to it. Default: current working directory.",
        envvar=EnvVar.WIKI_DIR,
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
    handle = log_run_start(wiki_dir=wiki_dir)
    _run_handle_holder["handle"] = handle


# Version management
def _get_version() -> str:
    """Get application version from package metadata."""
    try:
        return version("knox")
    except Exception:
        return "0.0.0"  # Fallback for development mode


def main() -> None:
    """Entry point for the CLI application."""
    _configure_logfire()
    configure_logging()
    _configure_streams_utf8()
    _register_for_argv(sys.argv[1:])
    handle: RunHandle | None = None
    try:
        app()
    except LookupError as exc:
        # Fail-fast vault validation (vault_paths): render cleanly instead of
        # dumping a traceback.
        handle = _extract_handle()
        get_err_console().print(f"error: {exc}")
        log_run_end(handle, status="error", error=repr(exc))
        raise SystemExit(1) from exc
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        handle = _extract_handle()
        log_run_end(handle, status="ok" if code == 0 else "error", exit_code=code)
        raise
    except BaseException as exc:  # noqa: BLE001 — capture any crash for the audit trail
        handle = _extract_handle()
        log_run_end(handle, status="error", error=repr(exc))
        raise
    handle = _extract_handle()
    log_run_end(handle, status="ok", exit_code=0)


def _extract_handle() -> RunHandle | None:
    """Retrieve the run handle stashed by the callback."""
    return _run_handle_holder["handle"]


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


def register_command(name: str) -> None:
    """Import the module that registers *name*.  No-op if unknown or already loaded."""
    module_name = _LAZY_MODULES.get(name)
    if module_name is not None:
        importlib.import_module(module_name)


def register_all_commands() -> None:
    """Import every command module (for ``--help``, bare ``knox``, shell completion)."""
    for module_name in dict.fromkeys(_LAZY_MODULES.values()):
        importlib.import_module(module_name)


def _find_requested_command(argv: list[str]) -> str | None:
    """Return the subcommand name from *argv*, or *None* if none is present.

    Handles global flags before the subcommand (``knox --wiki-dir x compile``).
    """
    for arg in argv:
        if arg in _LAZY_MODULES:
            return arg
    return None
