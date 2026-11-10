"""Harness adapter — builds a PydanticAI Agent for ACP serving.

Provides :func:`build_kompend_agent` (a generic agent with all kompend tools) and
:func:`kompend_session_config` (a per-session factory that roots tools at the
client's workspace and resolves the active agent definition).

Used by :mod:`kompend.cli.cmds.agent` to serve kompend over ACP via
``run_acp_stdio_sync``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, Caching, SelectModel
from pydantic_ai_harness.experimental.acp import AcpSession, AcpSessionConfig

from kompend.acp.definitions import AgentTask
from kompend.acp.model_selection import ConfigModelSelector
from kompend.acp.registry import AgentRegistry
from kompend.acp.tools import build_toolset
from kompend.core.config import resolve_config
from kompend.logging import get_logger

logger = get_logger(__name__)


def build_kompend_agent() -> Agent[None, str]:
    """Build the ACP session agent with no model of its own.

    pydantic-ai infers a string model eagerly in ``Agent.__init__``, so a
    placeholder here would either fail at construction or be sent to a provider
    that does not exist. The model comes from a ``SelectModel`` capability
    instead: it runs before every request step, so the server starts with
    nothing configured, and a session that configured an endpoint is served
    from the next turn on.

    The agent-level selector resolves against the process working directory;
    :func:`kompend_session_config` attaches the session-bound one, which wins
    (run-level capabilities are applied after the agent's own).
    """
    return Agent(
        None,
        instructions="You are a kompend knowledge base assistant.",
        name="acp-session-agent",
        capabilities=[SelectModel(ConfigModelSelector(Path.cwd()))],
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
        An ``AcpSessionConfig`` with deps=None, the resolved toolset, the
        session-bound model selector, and, unless ``compile.llm.prompt_cache``
        is off, pydantic-ai's ``Caching`` capability.
    """
    cwd = session.cwd
    registry = AgentRegistry(base_dir=cwd)
    definition = registry.active_for(AgentTask.GENERIC, {})

    toolset = build_toolset(
        base_dir=cwd,
        enabled=definition.tools.enabled,
    )

    # The session's own workspace decides which model serves it — an editor may
    # open several project roots through one server. Run-level capabilities are
    # applied after the agent's own, and the combined capability keeps the last
    # model contribution, so this selector outranks the agent-level fallback.
    capabilities: list[AbstractCapability[None]] = [SelectModel(ConfigModelSelector(Path(cwd)))]

    # A conversational session resends its whole history on every turn, which
    # is exactly what prompt caching pays off on. The capability is the
    # per-session form of the ``cache`` setting the agents use; the switch is
    # the same one (``compile.llm.prompt_cache``).
    cache_enabled = bool(resolve_config(base_dir=cwd).kompend.compile.llm.prompt_cache)
    if cache_enabled:
        capabilities.insert(0, Caching())

    return AcpSessionConfig(
        deps=None,
        toolsets=[toolset],
        capabilities=capabilities,
    )
