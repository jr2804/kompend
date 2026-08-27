"""``knox config`` — inspect and edit configuration."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import typer

from knox.cli._shared import vault_paths_from_ctx
from knox.cli.app import app
from knox.cli.args.common import JsonOutput
from knox.cli.args.config import ConfigGlobalOpt, ConfigKeyArg, ConfigValueArg
from knox.core.config import (
    STATUS_ERROR,
    STATUS_MODIFIED,
    STATUS_OK,
    STATUS_REMOVED,
    ConfigCheckEntry,
    ConfigCheckReport,
    check_config_file,
    check_config_files,
    coerce_config_value,
    get_global_config_file,
    get_project_config_file,
    resolve_config,
    set_config_file_value,
    validate_config_key,
)
from knox.core.config.keys import PATHS_WIKI_DIR
from knox.logging import get_console, get_err_console
from knox.types import JSON

config_app = typer.Typer(
    name="config",
    help="Inspect and edit KNOX configuration.",
    no_args_is_help=True,
)
app.add_typer(config_app)


_ENV_REF_RE = re.compile(r"^\$\{\w+\}$")


@config_app.command("get", help="Print a resolved config value.")
def config_get(
    ctx: typer.Context,
    key: ConfigKeyArg,
    global_: ConfigGlobalOpt = False,
) -> None:
    """Print the resolved value of KEY, respecting full precedence."""
    if not validate_config_key(key):
        get_err_console().print(f"error: unknown config key: {key}")
        raise typer.Exit(code=1)

    overrides = {}
    wiki_dir = vault_paths_from_ctx(ctx, require_vault=False).base_dir
    if wiki_dir:
        overrides[PATHS_WIKI_DIR] = str(wiki_dir)

    cfg = resolve_config(cli_overrides=overrides)
    value = cfg.get(key)
    get_console().print(_format_value(_redact_api_key(key, value)))


@config_app.command("set", help="Set a config value.")
def config_set(
    ctx: typer.Context,
    key: ConfigKeyArg,
    value: ConfigValueArg,
    global_: ConfigGlobalOpt = False,
) -> None:
    """Set KEY to VALUE in the project config (or global config with --global)."""
    if not validate_config_key(key):
        get_err_console().print(f"error: unknown config key: {key}")
        raise typer.Exit(code=1)

    try:
        coerced = coerce_config_value(value, key)
    except ValueError as exc:
        get_err_console().print(f"error: {exc}")
        raise typer.Exit(code=1) from exc

    if global_:
        path = get_global_config_file(create=True)
    else:
        wiki_dir = vault_paths_from_ctx(ctx, require_vault=False).base_dir
        path = get_project_config_file(start_dir=wiki_dir)
        if not path.exists():
            get_err_console().print("error: not in a KNOX vault — run `knox init` first or use `--global`")
            raise typer.Exit(code=1)

    try:
        set_config_file_value(path, key, coerced)
    except (OSError, ValueError) as exc:
        get_err_console().print(f"error: failed to write {path}: {exc}")
        raise typer.Exit(code=1) from exc

    get_console().print(f"set {key} = {_format_value(coerced)} in {path}")


@config_app.command("list", help="List all resolved config values with provenance.")
def config_list(
    ctx: typer.Context,
    global_: ConfigGlobalOpt = False,
) -> None:
    """Print every resolved config key, value, and source."""
    overrides = {}
    wiki_dir = vault_paths_from_ctx(ctx, require_vault=False).base_dir
    if wiki_dir:
        overrides[PATHS_WIKI_DIR] = str(wiki_dir)

    cfg = resolve_config(cli_overrides=overrides)
    for key, value in sorted(cfg.flat().items()):
        source = cfg.source_of(key)
        get_console().print(f"{key} = {_format_value(_redact_api_key(key, value))}  # {source}")


@config_app.command("path", help="Show the config file path.")
def config_file(
    ctx: typer.Context,
    global_: ConfigGlobalOpt = False,
) -> None:
    """Print the path to the project or global config file."""
    path = get_global_config_file(create=False) if global_ else get_project_config_file(start_dir=vault_paths_from_ctx(ctx, require_vault=False).base_dir)
    get_console().print(str(path), soft_wrap=True)


@config_app.command(
    "check",
    help="Diff config TOML files against the built-in defaults.",
)
def config_check(
    ctx: typer.Context,
    paths: list[Path] | None = typer.Argument(  # noqa: B008
        None,
        help="TOML files to diff. Omit to check the discovered project config. Multiple files merge (later wins); use --no-merge for per-file reports.",
    ),
    json_output: JsonOutput = False,
    no_merge: bool = typer.Option(  # noqa: B008
        False,
        "--no-merge",
        help="Report on each file independently instead of merging them.",
    ),
) -> None:
    """Show which keys are defaults, modified, or invalid for each file.

    Examples::

        knox config check                      # uses the discovered project config
        knox config check ./knox.toml
        knox config check ./base.toml ./local.toml         # merged
        knox config check ./base.toml ./local.toml --no-merge

    Exit code is 2 when any error (unknown key, validation failure) is found,
    1 when only modified keys are present, 0 otherwise. Combine with
    ``--json`` for machine-readable output.
    """
    if not paths:
        # No explicit file(s) → use the discovered project config (knox.toml
        # or .knox/config.toml). A vault with no config file at all is simply
        # all-defaults, so report that cleanly instead of 157 "file missing" rows.
        project_cfg = get_project_config_file(start_dir=vault_paths_from_ctx(ctx, require_vault=False).base_dir)
        if not project_cfg.exists():
            get_console().print(
                f"[yellow]No config file found at {project_cfg}[/yellow] — all values are at built-in defaults.",
            )
            raise typer.Exit(code=0)
        paths = [project_cfg]

    reports = [check_config_file(p) for p in paths] if no_merge or len(paths) == 1 else [check_config_files(paths)]

    if json_output:
        # Bypass Rich's line wrapping so JSON output is always valid even when
        # a Windows path string exceeds the terminal width.
        json.dump(
            [_report_payload(r) for r in reports],
            sys.stdout,
            indent=2,
            ensure_ascii=False,
        )
        sys.stdout.write("\n")
    else:
        for r in reports:
            _render_report_text(r)

    raise typer.Exit(code=_exit_code_for(reports))


def _report_payload(report: ConfigCheckReport) -> dict[str, JSON]:
    """Shape a report for JSON/agent consumption."""
    return report.to_dict()


def _render_report_text(report: ConfigCheckReport) -> None:
    """Render one report to the console as a status-keyed table."""
    console = get_console()
    if report.path is not None:
        console.print(f"[bold]config check[/bold] {report.path}")
    else:
        joined = ", ".join(str(p) for p in report.files_scanned)
        console.print(f"[bold]config check[/bold] (merged: {joined})")
    summary = f"  ok: [green]{report.ok_count}[/green]  modified: [yellow]{report.modified_count}[/yellow]"
    if report.removed_count:
        summary += f"  removed: [yellow]{report.removed_count}[/yellow]"
    summary += f"  error: [red]{report.error_count}[/red]"
    console.print(summary)
    console.print()

    # Show modified and error rows (skip ok rows to keep output compact).
    interesting = [e for e in report.entries if e.status != STATUS_OK]
    if not interesting:
        console.print("  (no modifications or errors)")
        return

    by_status: dict[str, list] = {
        STATUS_ERROR: [e for e in interesting if e.status == STATUS_ERROR],
        STATUS_REMOVED: [e for e in interesting if e.status == STATUS_REMOVED],
        STATUS_MODIFIED: [e for e in interesting if e.status == STATUS_MODIFIED],
    }
    colors = {STATUS_ERROR: "red", STATUS_REMOVED: "yellow", STATUS_MODIFIED: "yellow"}
    for label, entries in by_status.items():
        if not entries:
            continue
        color = colors[label]
        console.print(f"  [{color}]{label}[/{color}]:")
        for entry in entries:
            line = _format_entry_line(entry)
            console.print(f"    {line}")


def _format_entry_line(entry: ConfigCheckEntry) -> str:
    """Build one rendered line for a non-ok report entry."""
    if entry.status == STATUS_ERROR:
        msg = f"  ({entry.message})" if entry.message else ""
        current = _render_value(entry.current)
        return f"[red]ERROR[/red]   {entry.key}  current={current}{msg}"
    if entry.status == STATUS_REMOVED:
        msg = f"  ({entry.message})" if entry.message else ""
        current = _render_value(entry.current)
        return f"[yellow]REMOVED[/yellow] {entry.key}  current={current}{msg}"
    # modified
    current = _render_value(entry.current)
    default = _render_value(entry.default)
    return f"[yellow]MODIFIED[/yellow] {entry.key}  current={current}  default={default}"


def _render_value(value: JSON) -> str:
    """Format a value compactly for one-line display."""
    if value is None:
        return "<default>"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_render_value(v) for v in value) + "]"
    if isinstance(value, (dict,)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _exit_code_for(reports: list[ConfigCheckReport]) -> int:
    """Map a set of reports to a CLI exit code (0 clean, 1 modified/removed, 2 error)."""
    has_error = any(r.has_errors for r in reports)
    has_modified = any(r.modified_count > 0 for r in reports)
    has_removed = any(e.status == STATUS_REMOVED for r in reports for e in r.entries)
    if has_error:
        return 2
    if has_modified or has_removed:
        return 1
    return 0


def _format_value(value: JSON) -> str:
    """Format a config value for terminal output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _redact_api_key(key: str, value: JSON) -> JSON:
    """Hide provider API keys in CLI output; env-var refs (``${VAR}``) stay visible."""
    if key.rsplit(".", maxsplit=1)[-1] == "api_key" and isinstance(value, str) and not _ENV_REF_RE.match(value):
        return "***"
    return value
