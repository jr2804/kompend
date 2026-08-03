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
from rich.table import Table

from knox.core.reporting.runlog import RunHandle, log_run_end, log_run_start
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
    "assets": "knox.cli.cmds.knowledge.assets",
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
    "references": "knox.cli.cmds.knowledge.references",
    "review": "knox.cli.cmds.quality.review",
    "sources": "knox.cli.cmds.pipeline.sources",
    "mcp": "knox.cli.cmds.serve",
    "serve": "knox.cli.cmds.serve",
    "agent": "knox.cli.cmds.agent",
}

# Static one-line help per command, used ONLY by :func:`_print_fast_help` so
# ``knox --help`` renders instantly without importing command modules (which
# would pull pydantic-ai, the MCP SDK, and the parser backends). Keep this in
# sync with the ``help=`` text in each command module.
_COMMAND_HELP: dict[str, str] = {
    "agents": "Manage TOML-defined conversational agents.",
    "capture": "Extract knowledge items from text via LLM; save each as a source file.",
    "learn": "Store a single fact verbatim as a source file — no LLM involved.",
    "distill": "Merge captured notes into coherent wiki pages via LLM.",
    "write": "Create a wiki page synthesising content from one or more sources.",
    "doctor": "Validate configuration and connectivity for a wiki.",
    "status": "Show per-wiki statistics.",
    "list-parsers": "List all registered parsers (built-in + external + HTTP).",
    "trace": "Trace compiled wiki page(s) back to their source chunks.",
    "init": "Scaffold a new KNOX wiki project.",
    "log": "Regenerate the pipeline log from structured stores (no LLM calls).",
    "organize": "Categorise files in ``sources/`` into a hierarchical directory tree.",
    "pack-list": "List installed contribution packs.",
    "pack-install": "Install a built-in contribution pack.",
    "parse": "Parse documents and show their structure.",
    "compile": "Compile documents into a knowledge wiki.",
    "ingest": "Ingest documents into the knowledge graph without compiling a wiki.",
    "query": "Ask a natural language question against the compiled wiki.",
    "review": "Manage the review queue for compiled pages.",
    "mcp": "Start the KNOX MCP server over stdio transport.",
    "serve": "Start the KNOX web UI server.",
    "agent": "Run KNOX as an ACP agent on stdio.",
    "assets": "Manage source attachments (supplementary files).",
    "config": "Inspect and edit KNOX configuration.",
    "lint": "Lint a compiled wiki for quality issues and query lint history.",
    "memory": "Manage query memory and context synthesis.",
    "references": "Manage extracted bibliographic references.",
    "sources": "Manage ingested source files and their tags.",
}


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
    handle = log_run_start(wiki_dir=wiki_dir)
    _run_handle_holder["handle"] = handle


#: Module-level holder for the run handle, written by :func:`_callback`
#: (where Typer has resolved ``--wiki-dir``) and read by :func:`main`.
_run_handle_holder: dict[str, RunHandle | None] = {"handle": None}


def main() -> None:
    """Entry point for the CLI application."""
    configure_logging()
    _configure_streams_utf8()
    argv = sys.argv[1:]
    if _is_top_level_help(argv):
        _print_fast_help()
        return
    _register_for_argv(argv)
    handle: RunHandle | None = None
    try:
        app()
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


_run_handle_holder: dict[str, RunHandle | None] = {"handle": None}


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

    * ``--help`` / ``-h`` / empty argv → handled by :func:`_print_fast_help`
      before this is reached (no command modules are imported).
    * Shell-completion env vars (``*_COMPLETE``) → load all.
    * A known subcommand → load only its module.
    * Otherwise (global flags only, e.g. ``--version``) → load nothing.
    """
    if any(k.endswith("_COMPLETE") for k in os.environ):
        register_all_commands()
        return
    requested = _find_requested_command(argv)
    if requested is not None:
        register_command(requested)


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


def _is_top_level_help(argv: list[str]) -> bool:
    """True when *argv* asks for the top-level help (bare, ``--help``/``-h``).

    Returns ``False`` for a subcommand's own help (``knox compile --help``),
    which must still import that module to show its options.
    """
    if not argv:
        return True
    return any(a in ("--help", "-h") for a in argv) and _find_requested_command(argv) is None


def _print_fast_help() -> None:
    """Print top-level help instantly without importing any command module.

    Command names and one-line help come from the static ``_COMMAND_HELP``
    table, so ``knox --help`` avoids the heavy optional deps (pydantic-ai, MCP
    SDK, parser backends) that importing command modules would pull in.
    """
    console = get_console()
    console.print("[bold]Usage:[/bold] knox [OPTIONS] [COMMAND] [ARGS]...")
    console.print()
    console.print("Knowledge Organizer for eXtended documents")
    console.print()

    opts = Table(show_header=False, box=None, padding=(0, 2))
    opts.add_column(style="bold cyan")
    opts.add_column(style="bold cyan")
    opts.add_column()
    opts.add_column()
    for row in _GLOBAL_OPTIONS:
        opts.add_row(*row)
    console.print(opts)
    console.print()

    cmds = Table(show_header=False, box=None, padding=(0, 2))
    cmds.add_column(style="bold green")
    cmds.add_column()
    for name in sorted(_COMMAND_HELP):
        cmds.add_row(name, _COMMAND_HELP[name])
    console.print(cmds)


#: Static rows for the top-level options table in :func:`_print_fast_help`.
_GLOBAL_OPTIONS: list[tuple[str, str, str, str]] = [
    ("--version", "-v", "", "Show version and exit"),
    ("--wiki-dir", "-w", "TEXT", "Project root directory. All other paths are resolved relative to it. "
     "Default: current working directory. Env: KNOX_WIKI_DIR."),
    ("--no-user-tracking", "", "", "Do not record the OS user (ingested_by / memory_by) in metadata for this "
     "invocation. Env: KNOX_USER_TRACKING=false."),
    ("--install-completion", "", "", "Install completion for the current shell."),
    ("--show-completion", "", "", "Show completion for the current shell, to copy it or customize the installation."),
    ("--help", "", "", "Show this message and exit."),
]
