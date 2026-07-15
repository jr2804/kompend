"""ACP agent CLI — run KNOX as an ACP-compatible agent.

Usage::

    knox agent [--agent-file <path>] [--mcp-port <port>]

Starts an ACP agent on stdio that accepts ``providers/set`` for LLM
configuration and ``/knox <command>`` slash commands. When ``--mcp-port``
is set, also starts an MCP-over-HTTP server on the specified port so
connected AI agents can call KNOX's MCP tools directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys

import typer
from acp import run_agent

from knox.acp.knox_agent import KnoxAgent
from knox.cli.app import app
from knox.mcp.server import create_mcp_server


@app.command(name="agent")
def agent_cmd(
    agent_file: str | None = typer.Option(
        None,
        "--agent-file",
        help="Path to a TOML agent definition file to use instead of the configured active agent.",
    ),
    mcp_port: int | None = typer.Option(
        None,
        "--mcp-port",
        help="Port for the MCP-over-HTTP companion server (e.g. 8000). When set, KNOX starts both the ACP agent and an MCP HTTP server.",
    ),
) -> None:
    """Run KNOX as an ACP agent on stdio.

    Accepts ``providers/set`` for credential-free LLM configuration and
    ``/knox <command>`` slash commands for compilation, querying, etc.
    """
    asyncio.run(_run_agent(agent_file, mcp_port))


async def _run_agent(agent_file: str | None = None, mcp_port: int | None = None) -> None:
    """Start the ACP agent on stdio, optionally with an MCP companion server."""
    agent = KnoxAgent(agent_file=agent_file)

    # Start MCP companion server in background if port is specified
    mcp_task: asyncio.Task[None] | None = None
    if mcp_port is not None:
        mcp_server = create_mcp_server(port=mcp_port)
        mcp_task = asyncio.create_task(mcp_server.run_streamable_http_async())

    try:
        await run_agent(agent, sys.stdin.buffer, sys.stdout.buffer)
    finally:
        if mcp_task is not None:
            mcp_task.cancel()
            async with asyncio.timeout(5):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await mcp_task
