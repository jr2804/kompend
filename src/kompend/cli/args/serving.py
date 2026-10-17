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

# ── ACP agent / agents commands ─────────────────────────────────────────

AgentMcpPort = Annotated[
    int | None,
    typer.Option(
        "--mcp-port",
        help="Port for the MCP-over-HTTP companion server (e.g. 8000). When set, kompend starts both the ACP agent and an MCP HTTP server.",
    ),
]

AgentsAction = Annotated[
    str,
    typer.Argument(help="Action: list, use, validate, install, or remove"),
]

AgentsTask = Annotated[
    str | None,
    typer.Argument(help="Task name (for 'use'): compile, query, capture, organize"),
]

AgentsName = Annotated[
    str | None,
    typer.Argument(help="Agent name (for 'use') or file path (for 'validate')"),
]
