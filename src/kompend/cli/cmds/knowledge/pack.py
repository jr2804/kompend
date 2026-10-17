"""Pack management commands."""

from __future__ import annotations

import typer

from kompend.cli.app import app
from kompend.cli.args import PackName
from kompend.logging import get_console, get_err_console
from kompend.packs.base import get_registry
from kompend.packs.builtins import academic_pack, technical_pack


@app.command(name="pack-list")
def pack_list_cmd() -> None:
    """List installed contribution packs.

    Examples:
        kompend pack-list
    """
    registry = get_registry()
    if not registry.names:
        get_console().print("No packs registered.")
        return
    get_console().print(f"Registered packs ({len(registry.names)}):")
    for pack in registry.all:
        get_console().print(f"  {pack.name:<20} v{pack.version}  {pack.description}")


@app.command(name="pack-install")
def pack_install_cmd(
    name: PackName,
) -> None:
    """Install a built-in contribution pack.

    Examples:
        kompend pack-install academic
        kompend pack-install technical
    """
    registry = get_registry()
    pack_map = {
        "academic": academic_pack,
        "technical": technical_pack,
    }
    pack = pack_map.get(name.lower())
    if pack is None:
        available = ", ".join(sorted(pack_map))
        get_err_console().print(f"Unknown pack '{name}'. Available: {available}")
        raise typer.Exit(1)
    registry.register(pack)
    get_console().print(f"Installed pack '{pack.name}' v{pack.version}")
