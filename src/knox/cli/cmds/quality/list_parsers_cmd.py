"""``knox list-parsers`` — list all registered parsers."""

from __future__ import annotations

import json
import sys

import typer

from knox.cli.app import app
from knox.cli.args import JsonOutput
from knox.logging import get_console
from knox.preprocessing.parsers._registry import build_default_router, get_registered_parsers


@app.command(name="list-parsers")
def list_parsers_cmd(
    json_output: JsonOutput = False,
) -> None:
    """List all registered parsers (built-in + external + HTTP)."""
    build_default_router()

    parsers = get_registered_parsers()

    if json_output:
        json.dump(parsers, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    if not parsers:
        get_console().print("No parsers registered.")
        return

    get_console().print(f"Registered parsers ({len(parsers)}):")
    for p in parsers:
        exts = ", ".join(p.get("extensions", []) or [])
        ptype = p.get("type", "built-in")
        get_console().print(f"  \u2022 {p['name']} ({ptype})  [{exts}]", markup=False)
