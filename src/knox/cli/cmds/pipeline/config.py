"""``knox config`` — inspect and edit configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from knox.cli.app import app
from knox.cli.args.config import ConfigGlobalOpt, ConfigKeyArg, ConfigValueArg
from knox.core.config import (
    coerce_config_value,
    get_global_config_path,
    get_project_config_path,
    resolve_config,
    set_config_file_value,
    validate_config_key,
)
from knox.core.config.keys import PATHS_WIKI_DIR
from knox.logging import get_console, get_err_console

config_app = typer.Typer(
    name="config",
    help="Inspect and edit KNOX configuration.",
    no_args_is_help=True,
)
app.add_typer(config_app)


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
    wiki_dir = _resolve_wiki_dir(ctx)
    if wiki_dir:
        overrides[PATHS_WIKI_DIR] = str(wiki_dir)

    cfg = resolve_config(cli_overrides=overrides)
    value = cfg.get(key)
    get_console().print(_format_value(value))


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
        path = get_global_config_path(create=True)
    else:
        wiki_dir = _resolve_wiki_dir(ctx)
        path = get_project_config_path(start_dir=wiki_dir)
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
    wiki_dir = _resolve_wiki_dir(ctx)
    if wiki_dir:
        overrides[PATHS_WIKI_DIR] = str(wiki_dir)

    cfg = resolve_config(cli_overrides=overrides)
    for key, value in sorted(cfg.flat().items()):
        source = cfg.source_of(key)
        get_console().print(f"{key} = {_format_value(value)}  # {source}")


@config_app.command("path", help="Show the config file path.")
def config_path(
    ctx: typer.Context,
    global_: ConfigGlobalOpt = False,
) -> None:
    """Print the path to the project or global config file."""
    path = get_global_config_path(create=False) if global_ else get_project_config_path(start_dir=_resolve_wiki_dir(ctx))
    get_console().print(str(path), soft_wrap=True)


def _resolve_wiki_dir(ctx: typer.Context) -> Path | None:
    """Resolve --wiki-dir from callback context, if set."""
    raw = ctx.obj.get("wiki_dir") if ctx.obj else None
    return Path(raw).resolve() if raw else None


def _format_value(value: Any) -> str:
    """Format a config value for terminal output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)
