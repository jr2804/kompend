"""``knox config`` — inspect and edit configuration."""

from __future__ import annotations

from typing import Any

import typer

from knox.cli.app import app
from knox.cli.args_config import ConfigGlobalOpt, ConfigKeyArg, ConfigValueArg
from knox.core.config import (
    coerce_config_value,
    get_global_config_path,
    get_project_config_path,
    resolve_config,
    set_config_file_value,
    validate_config_key,
)

config_app = typer.Typer(
    name="config",
    help="Inspect and edit KNOX configuration.",
    no_args_is_help=True,
)
app.add_typer(config_app)


@config_app.command("get", help="Print a resolved config value.")
def config_get(
    key: ConfigKeyArg,
    global_: ConfigGlobalOpt = False,
) -> None:
    """Print the resolved value of KEY, respecting full precedence."""
    if not validate_config_key(key):
        typer.echo(f"error: unknown config key: {key}", err=True)
        raise typer.Exit(code=1)

    cfg = resolve_config()
    value = cfg.get(key)
    typer.echo(_format_value(value))


@config_app.command("set", help="Set a config value.")
def config_set(
    key: ConfigKeyArg,
    value: ConfigValueArg,
    global_: ConfigGlobalOpt = False,
) -> None:
    """Set KEY to VALUE in the project config (or global config with --global)."""
    if not validate_config_key(key):
        typer.echo(f"error: unknown config key: {key}", err=True)
        raise typer.Exit(code=1)

    try:
        coerced = coerce_config_value(value, key)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    path = get_global_config_path(create=True) if global_ else get_project_config_path(create=True)

    try:
        set_config_file_value(path, key, coerced)
    except (OSError, ValueError) as exc:
        typer.echo(f"error: failed to write {path}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"set {key} = {_format_value(coerced)} in {path}")


@config_app.command("list", help="List all resolved config values with provenance.")
def config_list(
    global_: ConfigGlobalOpt = False,
) -> None:
    """Print every resolved config key, value, and source."""
    cfg = resolve_config()
    for key, value in sorted(cfg.flat().items()):
        source = cfg.source_of(key)
        typer.echo(f"{key} = {_format_value(value)}  # {source}")


@config_app.command("path", help="Show the config file path.")
def config_path(
    global_: ConfigGlobalOpt = False,
) -> None:
    """Print the path to the project or global config file."""
    path = get_global_config_path(create=False) if global_ else get_project_config_path(create=False)
    typer.echo(str(path))


def _format_value(value: Any) -> str:
    """Format a config value for terminal output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)
