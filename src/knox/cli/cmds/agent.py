"""ACP agent CLI — run KNOX as an ACP-compatible agent.

Usage::

    knox agent [--agent-file <path>]

Starts an ACP agent on stdio that accepts ``providers/set`` for LLM
configuration and ``/knox <command>`` slash commands.
"""

from __future__ import annotations

import asyncio
import sys

import typer

from knox.agent.knox_agent import KnoxAgent
from knox.cli.app import app


@app.command(name="agent")
def agent_cmd(
    agent_file: str | None = typer.Option(
        None,
        "--agent-file",
        help="Path to a TOML agent definition file to use instead of the configured active agent.",
    ),
) -> None:
    """Run KNOX as an ACP agent on stdio.

    Accepts ``providers/set`` for credential-free LLM configuration and
    ``/knox <command>`` slash commands for compilation, querying, etc.
    """
    asyncio.run(_run_agent(agent_file))


async def _run_agent(agent_file: str | None = None) -> None:
    """Start the ACP agent on stdio."""
    from acp import run_agent  # noqa: PLC0415 — lazy import, acp is heavy

    agent = KnoxAgent(agent_file=agent_file)
    await run_agent(agent, sys.stdin.buffer, sys.stdout.buffer)
