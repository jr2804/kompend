"""Pack management commands."""

from __future__ import annotations

import typer

from knox.cli.app import app
from knox.packs.base import get_registry
from knox.packs.builtins import academic_pack, technical_pack


@app.command(name="pack-list")
def pack_list_cmd() -> None:
    """List installed contribution packs.

    Examples:
        knox pack-list
    """
    registry = get_registry()
    if not registry.names:
        typer.echo("No packs registered.")
        return
    typer.echo(f"Registered packs ({len(registry.names)}):")
    for pack in registry.all:
        typer.echo(f"  {pack.name:<20} v{pack.version}  {pack.description}")


@app.command(name="pack-install")
def pack_install_cmd(
    name: str = typer.Argument(..., help="Pack name to install (e.g. 'academic')."),
) -> None:
    """Install a built-in contribution pack.

    Examples:
        knox pack-install academic
        knox pack-install technical
    """
    registry = get_registry()
    pack_map = {
        "academic": academic_pack,
        "technical": technical_pack,
    }
    pack = pack_map.get(name.lower())
    if pack is None:
        available = ", ".join(sorted(pack_map))
        typer.echo(f"Unknown pack '{name}'. Available: {available}", err=True)
        raise typer.Exit(1)
    registry.register(pack)
    typer.echo(f"Installed pack '{pack.name}' v{pack.version}")
