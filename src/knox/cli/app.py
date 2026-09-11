"""Typer CLI app.

Commands are registered lazily through :class:`_LazyGroup` (Click's
native ``list_commands``/``get_command`` pattern, adapted to Typer), so
Typer owns the full command tree — help, completion, and error
suggestions work natively in every position (``knox --help``,
``knox -w <dir> --help``, bare ``knox``) — while only the invoked
subcommand's module is actually imported. Command modules keep heavy
dependencies inside their command bodies so per-command startup stays
fast.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from importlib.metadata import version
from typing import Final

import click
import logfire
import typer
from typer.core import TyperGroup

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

#: Subcommand name → module path. Importing the module registers its
#: command(s) on ``app`` via the ``@app.command()`` decorator side effect.
#: Multiple subcommands can share one module (e.g. ``compile`` + ``ingest``
#: + ``parse`` all live in ``pipeline.py``). Consumed by :class:`_LazyGroup`.
_COMMAND_MODULES: Final[dict[str, str]] = {
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


class _LazyGroup(TyperGroup):
    """Click group that imports command modules on demand.

    This is Click's documented lazy-subcommand pattern (``list_commands`` /
    ``get_command``), adapted to Typer: Typer pre-builds command objects
    when ``@app.command()`` decorates a function, so the group can't hold
    them until the module is imported. ``list_commands`` reports every
    registered name without importing anything (fast ``--help`` naming),
    and ``get_command`` imports the module and attaches the freshly built
    click command to this group.

    Framework-native: no argv sniffing — Typer/Click decide when commands
    are needed (dispatch, help rendering, completion, suggestions).
    """

    def list_commands(self, ctx: click.Context) -> list[str]:  # noqa: PLR6301 — Click API override
        return sorted(_COMMAND_MODULES)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name in self.commands:
            return self.commands[cmd_name]
        module_name = _COMMAND_MODULES.get(cmd_name)
        if module_name is None:
            return None
        importlib.import_module(module_name)  # registers commands / groups on app
        from typer.main import get_command_from_info, get_group  # noqa: PLC0415 — typer internal, cheap

        # Plain commands from @app.command()
        for info in app.registered_commands:
            cmd = get_command_from_info(
                info,
                pretty_exceptions_short=True,
                rich_markup_mode=None,
            )
            if cmd.name == cmd_name:
                self.add_command(cmd)
                return cmd
        # Sub-Typer groups from app.add_typer() (config, lint, attachments, …).
        # Aliases ("att" → attachments_app) live on the GroupInfo name.
        for group_info in app.registered_groups:
            gname = group_info.name or group_info.typer_instance.info.name or ""
            if gname == cmd_name:
                sub_group = get_group(group_info.typer_instance)
                sub_group.name = gname
                self.add_command(sub_group)
                return sub_group
        return None


app = typer.Typer(
    name="knox",
    help="Knowledge Organizer for eXtended documents",
    add_completion=True,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    cls=_LazyGroup,
)


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


def register_all_commands() -> None:
    """Import every command module (test-support introspection helper).

    Dispatch never needs this — :class:`_LazyGroup` imports on demand.
    Tests that enumerate the full command surface (CLI↔MCP parity) call
    it so ``app.registered_commands`` is fully populated.
    """
    for module_name in dict.fromkeys(_COMMAND_MODULES.values()):
        importlib.import_module(module_name)
from typer.main import get_command_from_info, get_group
