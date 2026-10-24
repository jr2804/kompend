"""Typer CLI app.

Commands come from :mod:`kompend.cli._registry` and are registered lazily by
:class:`kompend.cli._group.LazyGroup`, so help names every command while only
the invoked subcommand's module is imported. Command modules keep heavy
dependencies inside their command bodies so per-command startup stays fast.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from importlib.metadata import version

import typer

from kompend.cli._group import LazyGroup
from kompend.cli._registry import COMMANDS, OnDemandModule
from kompend.cli._shared import format_exception_for_cli
from kompend.cli.args import ConfigPath, NoUserTracking, Verbose, Version, WikiDir
from kompend.core.cancel import CANCEL_EXIT_CODE, KompendCancelled
from kompend.core.reporting.runlog import RunHandle, log_run_end, log_run_start
from kompend.core.verbosity import is_verbose, set_verbose
from kompend.logging import configure_logging, get_console

app = typer.Typer(
    name="kompend",
    help="Compile a document corpus into an interlinked, OKF-conformant markdown wiki",
    add_completion=True,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    cls=LazyGroup,
)

LazyGroup.bind(app)


#: Module-level holder for the run handle, written by :func:`_callback`
#: (where Typer has resolved ``--wiki-dir``) and read by :func:`main`.
_run_handle_holder: dict[str, RunHandle | None] = {"handle": None}


@app.callback(invoke_without_command=True)
def _callback(
    ctx: typer.Context,
    version: Version = False,
    wiki_dir: WikiDir = None,
    config: ConfigPath = None,
    no_user_tracking: NoUserTracking = False,
    verbose: Verbose = False,
) -> None:
    """Compile a document corpus into an interlinked, OKF-conformant markdown wiki"""
    set_verbose(verbose)
    if version:
        get_console().print(_get_version())
        raise typer.Exit()
    ctx.obj = {"wiki_dir": wiki_dir, "config_file": config, "no_user_tracking": no_user_tracking}
    # Framework-parsed tokens: the group parser has already consumed its own
    # options (--wiki-dir, --version, ...), so protected_args + args is
    # exactly subcommand + its raw args. No argv sniffing.
    cmd_tokens = [*ctx._protected_args, *ctx.args]
    handle = log_run_start(wiki_dir=wiki_dir, cmd_tokens=cmd_tokens)
    _run_handle_holder["handle"] = handle


# Version management
def _get_version() -> str:
    """Get application version from package metadata."""
    try:
        return version("kompend")
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
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        handle = _extract_handle()
        log_run_end(handle, status="ok" if code == 0 else "error", exit_code=code)
        raise
    except KompendCancelled as exc:
        # Graceful user abort (double-ESC during a progress bar): the command
        # already saved its progress and unwound cleanly — report and exit
        # 130, no traceback.
        handle = _extract_handle()
        log_run_end(handle, status="cancelled", exit_code=CANCEL_EXIT_CODE, error=exc.reason)
        sys.stderr.write(f"\nCancelled ({exc.reason}) — progress saved; rerun the command to resume.\n")
        raise SystemExit(CANCEL_EXIT_CODE) from exc
    except KeyboardInterrupt:
        # System-level interrupt — keep default Python behaviour (no reformatting).
        handle = _extract_handle()
        log_run_end(handle, status="error", error="KeyboardInterrupt")
        raise
    except BaseException as exc:  # noqa: BLE001 — capture any crash for the audit trail
        handle = _extract_handle()
        log_run_end(handle, status="error", error=repr(exc))
        sys.stderr.write(format_exception_for_cli(exc, verbose=is_verbose()) + "\n")
        raise SystemExit(1) from exc
    handle = _extract_handle()
    log_run_end(handle, status="ok", exit_code=0)


# Logfire instrumentation: configure() must run before instrument_pydantic_ai().
# We point at "kompend" + "staging" so traces land in the starter project on
# logfire-eu. Respects KOMP_LOGFIRE_DISABLED and LOGFIRE_TOKEN.
def _configure_logfire() -> None:
    """Enable Logfire instrumentation once per process (called from main()).

    kompend.telemetry owns ``import logfire`` (~700 ms, and it pulls in
    requests) and is loaded here on demand, the same way the lazy command group
    loads command modules. The alternative - importing it at module scope - made
    every in-process import of this module, including ``kompend --version``,
    ``--help`` and the whole test suite, pay for telemetry none of them emit.
    """
    telemetry = importlib.import_module(OnDemandModule.TELEMETRY.value)
    telemetry.configure_logfire()


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

    Dispatch never needs this — :class:`~kompend.cli._group.LazyGroup` imports on demand.
    Tests that enumerate the full command surface (CLI↔MCP parity) call
    it so ``app.registered_commands`` is fully populated.
    """
    for module_name in sorted({spec.module.value for spec in COMMANDS.values()}):
        importlib.import_module(module_name)
