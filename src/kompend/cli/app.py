"""The kompend CLI application.

Commands come from :mod:`kompend.cli.tree.registry` and are registered as import
paths, so ``kompend --help`` names every command (and renders each panel) while
only the invoked command's module is imported. Command modules keep heavy
dependencies inside their command bodies so per-command startup stays fast.

The root options (``--wiki-dir``, ``--config``, ``--no-user-tracking``,
``--verbose``, ``--version``) live on the meta app: they are parsed before a
command runs and published to :mod:`kompend.cli.context`, which is how
commands read them without a framework context object in every signature.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, Literal

from cyclopts import App, Group, Parameter

from kompend.cli.args import ConfigPath, NoUserTracking, Verbose, WikiDir
from kompend.cli.context import RunContext, use_run_context
from kompend.cli.errors import format_exception_for_cli
from kompend.cli.tree.registry import COMMANDS, PANEL_ORDER, HelpPanel, OnDemandModule
from kompend.core.cancel import CANCEL_EXIT_CODE, KompendCancelled
from kompend.core.reporting.handle import RunHandle
from kompend.core.verbosity import is_verbose, set_verbose
from kompend.logging import apply_logging_config, configure_logging, get_console, route_third_party_loggers

Version = Annotated[
    bool,
    Parameter(
        name=("--version", "-v"),
        help="Show version and exit",
        negative="",
    ),
]

app = App(
    name="kompend",
    help="Compile a document corpus into an interlinked, OKF-conformant markdown wiki",
)
#: Help flags carry no ``-h`` short form: ``-h`` stays an unknown option so
#: mistyped short flags fail loudly instead of silently printing help.
app.help_flags = ["--help"]


class _RunHandleSlot:
    """Single-value holder for the run-log handle of the invocation in flight."""

    __slots__ = ("handle",)

    def __init__(self) -> None:
        self.handle: RunHandle | None = None


#: Single-value holder for the run handle: written by the meta command (where
#: ``--wiki-dir`` has been resolved) and read by :func:`main`.
_run_handle_slot: _RunHandleSlot = _RunHandleSlot()


#: One ``Group`` instance per help panel — Cyclopts dedupes groups by identity,
#: so building a new ``Group`` per command collides.
_PANEL_GROUPS: dict[HelpPanel, Group] = {panel: Group(name=panel.value, sort_key=PANEL_ORDER.index(panel)) for panel in PANEL_ORDER}


def _panel_group(panel: HelpPanel) -> Group:
    """Render a help panel in workflow order, not alphabetical order."""
    return _PANEL_GROUPS[panel]


for _name, _spec in COMMANDS.items():
    app.command(
        f"{_spec.module.value}:{_spec.entry}",
        name=_name.value,
        help=_spec.short_help,
        group=_panel_group(_spec.panel),
    )

# Cyclopts registers help/version as synthetic commands; they belong in the
# parameter list, not in the command panels (they are flags, not commands).
for _synthetic in ("--help", "-h", "--version"):
    if _synthetic in app:
        app[_synthetic].show = False

# Shell completion: Cyclopts ships the install command; ``--show-completion``
# prints the same script so the previous CLI surface keeps working.
app.register_install_completion_command(
    name="--install-completion",
    group=_panel_group(HelpPanel.SETUP),
    help="Install shell completion for the detected shell.",
)

CompletionShell = Annotated[
    Literal["zsh", "bash", "fish", "powershell"] | None,
    Parameter(name="--shell", help="Shell to print completion for (default: detect the current shell)."),
]


@app.command(
    name="--show-completion",
    group=_panel_group(HelpPanel.SETUP),
    help="Print the shell completion script for the detected shell.",
)
def _show_completion(*, shell: CompletionShell = None) -> None:
    """Write the completion script to stdout (redirect it into your shell's rc)."""
    get_console().print(app.generate_completion(shell=shell))


@app.meta.default
def _root(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
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
        raise SystemExit(0)
    # Telemetry configures here, not in main(): the version early exit above must
    # not pay the ~700 ms logfire import. Cached, so several in-process
    # invocations configure once.
    _configure_logfire()
    context = RunContext(wiki_dir=wiki_dir, config_file=config, no_user_tracking=no_user_tracking)
    with use_run_context(context):
        # knox-nnm6: the operator's [logging] modules apply before any command
        # body runs. Commands re-resolve config for their own values; this key
        # has no flag, so the file-level resolution here is final for it.
        _apply_logging_config(wiki_dir, config)
        handle = log_run_start(wiki_dir=wiki_dir, cmd_tokens=list(tokens))
        _run_handle_slot.handle = handle
        app(list(tokens))


# Version management
@cache
def _apply_logging_config(wiki_dir: str | None, config_file: str | None) -> None:
    """Apply the operator's ``[logging] modules`` overrides (knox-nnm6).

    The config engine and path resolver load behind :func:`importlib.import_module`
    on purpose: ``kompend --version`` must stay light, the same posture as
    :func:`_get_version` below.
    """
    config_module = importlib.import_module("kompend.core.config")
    paths_module = importlib.import_module("kompend.core.paths")
    base_dir = paths_module.vault_paths(wiki_dir, require_vault=False).base_dir
    apply_logging_config(config_module.resolve_config(base_dir=base_dir, config_file=config_file))


def _get_version() -> str:
    """Installed version of kompend, read from package metadata on first use.

    ``importlib.metadata`` pulls in ``email`` and friends (~200 ms); the only
    caller is ``--version``, so it must not load for every command.
    """
    metadata = importlib.import_module("importlib.metadata")
    try:
        return metadata.version("kompend")
    except metadata.PackageNotFoundError:
        return "0.0.0"


def main() -> None:
    """Entry point for the CLI application."""
    configure_logging()
    route_third_party_loggers()
    _configure_streams_utf8()
    handle: RunHandle | None = None
    try:
        app.meta()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        handle = _extract_handle()
        _end_run(handle, status="ok" if code == 0 else "error", exit_code=code)
        _print_run_log_footer(handle)
        raise
    except KompendCancelled as exc:
        # Graceful user abort (double-ESC during a progress bar): the command
        # already saved its progress and unwound cleanly — report and exit
        # 130, no traceback.
        handle = _extract_handle()
        _end_run(handle, status="cancelled", exit_code=CANCEL_EXIT_CODE, error=exc.reason)
        _write_failure(f"\nCancelled ({exc.reason}) — progress saved; rerun the command to resume.")
        _print_run_log_footer(handle)
        raise SystemExit(CANCEL_EXIT_CODE) from exc
    except KeyboardInterrupt:
        # System-level interrupt — keep default Python behaviour (no reformatting).
        handle = _extract_handle()
        _end_run(handle, status="error", error="KeyboardInterrupt")
        _print_run_log_footer(handle)
        raise
    except BaseException as exc:  # noqa: BLE001 — capture any crash for the audit trail
        handle = _extract_handle()
        _end_run(handle, status="error", error=repr(exc))
        _write_failure(format_exception_for_cli(exc, verbose=is_verbose()))
        _print_run_log_footer(handle)
        raise SystemExit(1) from exc
    handle = _extract_handle()
    _end_run(handle, status="ok", exit_code=0)
    _print_run_log_footer(handle)


def _print_run_log_footer(handle: RunHandle | None) -> None:
    """Name the flight-recorder file when this run wrote one.

    One line after the command's own summary, so a crash report can point at
    the full detail without the user reproducing it with ``--verbose``. Silent
    when no file was written (no vault, or an unwritable log directory).
    """
    path = handle.log_file if handle is not None else None
    if path is None or not path.is_file():
        return
    try:
        shown = path.relative_to(Path.cwd())
    except ValueError:
        shown = path
    get_console().print(f"Full log: {shown}", soft_wrap=True)


def _end_run(handle: RunHandle | None, **fields: Any) -> None:
    """Close the run record; a broken audit trail must not hide the outcome."""
    try:
        log_run_end(handle, **fields)
    except BaseException:  # noqa: BLE001 - the caller is already on an error path
        return


def _write_failure(text: str) -> None:
    """Print a failure so the user sees it, whatever the console is doing.

    A dead console handle (``OSError: [WinError 6] The handle is invalid``, seen
    when a Windows terminal loses its pipe mid-run) makes ``sys.stderr.write``
    raise, which is how a failing command can exit with no message at all. Each
    fallback is cheaper than the last and needs less of the process to work.
    """
    payload = text if text.endswith("\n") else f"{text}\n"
    try:
        sys.stderr.write(payload)
        sys.stderr.flush()
        return
    except BaseException:  # noqa: BLE001, S110 - a dead console is the case this exists for
        pass
    with contextlib.suppress(BaseException):
        os.write(2, payload.encode("utf-8", "replace"))


# Logfire instrumentation: configure() must run before instrument_pydantic_ai().
# We point at "kompend" + "staging" so traces land in the starter project on
# logfire-eu. Respects KOMP_LOGFIRE_DISABLED and LOGFIRE_TOKEN.
@cache
def _configure_logfire() -> None:
    """Enable Logfire instrumentation once per process (called from the meta command).

    kompend.telemetry owns ``import logfire`` (~700 ms, and it pulls in
    requests) and is loaded here on demand, the same way command modules are
    loaded on demand. The call lives in :func:`_root` — behind the ``--version``
    early exit, and after ``--help`` has already been served — so bare
    invocations that emit no telemetry never import it. ``@cache`` keeps a
    shared process (tests, MCP) at one import and one configure.
    """
    telemetry = importlib.import_module(OnDemandModule.TELEMETRY.value)
    telemetry.configure_logfire()


def log_run_start(
    wiki_dir: str | None = None,
    *,
    cmd_tokens: list[str] | None = None,
) -> RunHandle | None:
    """Record that a command started (no-op without a vault)."""
    handle: RunHandle | None = _runlog().log_run_start(wiki_dir=wiki_dir, cmd_tokens=cmd_tokens)
    return handle


def log_run_end(
    handle: RunHandle | None,
    *,
    status: str,
    exit_code: int | None = None,
    error: str | None = None,
) -> None:
    """Record how a command ended; nothing to write without a handle."""
    if handle is None:
        return
    _runlog().log_run_end(handle, status=status, exit_code=exit_code, error=error)


@cache
def _runlog() -> ModuleType:
    """The run-log module, imported the first time a real command runs.

    ``--version`` and ``--help`` exit before any run is recorded, and the run
    log imports the config engine and the vault store, so it must not load at
    module scope either.
    """
    return importlib.import_module(OnDemandModule.RUNLOG.value)


def _extract_handle() -> RunHandle | None:
    """Retrieve the run handle stashed by the meta command."""
    return _run_handle_slot.handle


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
    """Resolve every command so its module is imported (test-support helper).

    Dispatch never needs this — the app resolves an import path on first use.
    Tests that enumerate the full command surface (CLI↔MCP parity) call it so
    every command's parameters are bound.
    """
    for name in COMMANDS:
        app[name]
