"""ACP agent CLI — run KNOX as an ACP-compatible agent.

Usage::

    knox agent [--mcp-port <port>]

Starts an ACP agent on stdio via ``pydantic-ai-harness``'s
``run_acp_stdio_sync``. The harness handles the ACP protocol, session
management, streaming, tool approval, and cancellation. KNOX provides
the agent, tools, and per-session configuration.
"""

from __future__ import annotations

import asyncio

import typer
from pydantic_ai_harness.experimental.acp import run_acp_stdio_sync

from knox.acp.harness_adapter import build_knox_agent, knox_session_config
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

    # Start MCP companion server in background if port is specified
    if mcp_port is not None:
        mcp_server = create_mcp_server(port=mcp_port)
        mcp_task = asyncio.create_task(mcp_server.run_streamable_http_async())
        try:
            run_acp_stdio_sync(
                agent,
                session_config=knox_session_config,
                name="knox",
                version="0.1.0",
            )
        finally:
            mcp_task.cancel()
            try:
                asyncio.get_event_loop().run_until_complete(
                    asyncio.wait_for(mcp_task, timeout=5)
                )
            except (asyncio.CancelledError, Exception, TimeoutError):
                pass
    else:
        run_acp_stdio_sync(
            agent,
            session_config=knox_session_config,
            name="knox",
            version="0.1.0",
        )
