"""CLI arguments for the serving commands (serve, mcp, agent)."""

from __future__ import annotations

from typing import Annotated

import typer

# ── N5 web serve command ─────────────────────────────────────────────────

ServeHost = Annotated[
    str,
    typer.Option(
        "--host",
        help="Host to bind the web server to. Default: 127.0.0.1 (localhost only).",
    ),
]

ServePort = Annotated[
    int,
    typer.Option(
        "--port",
        "-p",
        help="Port to bind the web server to.",
        min=1024,
        max=65535,
    ),
]

ServeOpen = Annotated[
    bool,
    typer.Option(
        "--open",
        help="Open the web UI in the default browser automatically.",
    ),
]
