"""Harness adapter — builds a PydanticAI Agent for ACP serving.

Provides :func:`build_kompend_agent` (a generic agent with all kompend tools) and
:func:`kompend_session_config` (a per-session factory that roots tools at the
client's workspace and resolves the active agent definition).

Used by :mod:`kompend.cli.cmds.agent` to serve kompend over ACP via
``run_acp_stdio_sync``.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai_harness.experimental.acp import AcpSession, AcpSessionConfig

from kompend.acp.definitions import AgentTask
from kompend.acp.registry import AgentRegistry
from kompend.acp.tools import build_toolset
from kompend.logging import get_logger

logger = get_logger(__name__)


def build_kompend_agent() -> Agent[None, str]:
    """Build a generic PydanticAI Agent with a placeholder model.

    The actual model is resolved per-session via the config or
    ``providers/set`` override. The agent uses a minimal system prompt —
    the per-session ``session_config`` factory overrides instructions
    based on the active agent definition.
    """
    return Agent(
        "noop",
        instructions="You are a kompend knowledge base assistant.",
        name="acp-session-agent",
    )


def kompend_session_config(session: AcpSession) -> AcpSessionConfig[None]:
    """Per-session factory for ``run_acp_stdio_sync(session_config=...)``.

    Resolves the active agent definition for the session's workspace and
    returns an ``AcpSessionConfig`` with the appropriate toolset and
    instructions.

    Args:
        session: The :class:`AcpSession` from the harness (has ``cwd``,
            ``mcp_servers``, ``client_capabilities``).

    Returns:
        An ``AcpSessionConfig`` with deps=None and the resolved toolset.
    """
    cwd = session.cwd
    registry = AgentRegistry(base_dir=cwd)
    definition = registry.active_for(AgentTask.GENERIC, {})

    toolset = build_toolset(
        base_dir=cwd,
        enabled=definition.tools.enabled,
    )

    return AcpSessionConfig(
        deps=None,
        toolsets=[toolset],
    )
