"""ACP agent CLI — run KNOX as an ACP-compatible agent.

Usage::

    knox agent [--mcp-port <port>]

Starts an ACP agent on stdio via ``pydantic-ai-harness``'s
``run_acp_stdio_sync``. The harness handles the ACP protocol, session
management, streaming, tool approval, and cancellation. KNOX provides
the agent, tools, and per-session configuration.

Heavy imports (the harness ACP machinery, the MCP server factory) live
inside the command body: this module is imported eagerly by the CLI so
Typer can list every command in help — it must stay cheap to import.
"""

from __future__ import annotations

import asyncio
import contextlib

import typer
from pydantic_ai import Agent
from pydantic_ai_harness.experimental.acp import run_acp_stdio_sync

from knox.acp import build_knox_agent, knox_session_config
from knox.cli.app import app
from knox.mcp.server import create_mcp_server


@app.command(name="agent")
def agent_cmd(
    mcp_port: int | None = typer.Option(
        None,
        "--mcp-port",
        help="Port for the MCP-over-HTTP companion server (e.g. 8000). When set, KNOX starts both the ACP agent and an MCP HTTP server.",
    ),
) -> None:
    """Run KNOX as an ACP agent on stdio.

    Serves a PydanticAI agent over the Agent Client Protocol using
    ``pydantic-ai-harness``. The agent exposes KNOX tools (compile,
    query, capture, config) and resolves per-session configuration
    from the workspace the editor opened.
    """
    agent = build_knox_agent()

    if mcp_port is not None:
        mcp_server = create_mcp_server()
        mcp_task = asyncio.create_task(mcp_server.run_http_async(port=mcp_port, transport="streamable-http"))
        try:
            _run_sync(agent)
        finally:
            mcp_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception, TimeoutError):
                asyncio.get_event_loop().run_until_complete(asyncio.wait_for(mcp_task, timeout=5))
    else:
        _run_sync(agent)


def _run_sync(agent: Agent[None, str]) -> None:
    """Run the ACP agent synchronously."""
    run_acp_stdio_sync(
        agent,
        session_config=knox_session_config,
        name="knox",
        version="0.1.0",
    )
