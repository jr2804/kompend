"""ACP agent CLI — run KNOX as an ACP-compatible agent.

Usage::

    knox agent

Starts an ACP agent on stdio that accepts ``providers/set`` for LLM
configuration and ``/knox <command>`` slash commands.
"""

from __future__ import annotations

import asyncio
import sys

from knox.agent.knox_agent import KnoxAgent


def agent_cmd() -> None:
    """Run KNOX as an ACP agent on stdio.

    Accepts ``providers/set`` for credential-free LLM configuration and
    ``/knox <command>`` slash commands for compilation, querying, etc.
    """
    asyncio.run(_run_agent())


async def _run_agent() -> None:
    """Start the ACP agent on stdio."""
    from acp import run_agent  # noqa: PLC0415 — lazy import, acp is heavy

    agent = KnoxAgent()
    await run_agent(agent, sys.stdin.buffer, sys.stdout.buffer)
