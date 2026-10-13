"""Discovery, loading, and selection of TOML-defined conversational agents.

Agents are discovered from three sources (later wins):

1. **Built-in** — ``src/kompend/acp/builtin/*.toml`` (shipped with Kompend)
2. **Vault** — ``<base_dir>/.kompend/agents/*.toml`` (user-installed)
3. **Explicit** — ``--agent-file path`` (CLI override, not yet implemented)

The active agent per task is stored in ``config.toml`` under ``[agents]``
and resolved by :meth:`AgentRegistry.active_for`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kompend.acp.definitions import AgentDefinition, AgentLLMConfig, AgentPromptConfig, AgentTask
from kompend.core.config.keys import (
    AGENTS_CAPTURE,
    AGENTS_COMPILE,
    AGENTS_GENERIC,
    AGENTS_ORGANIZE,
    AGENTS_QUERY,
)
from kompend.logging import get_logger

_BUILTIN_DIR = Path(__file__).resolve().parent / "builtin"

_TASK_CONFIG_KEY: dict[AgentTask, str] = {
    AgentTask.COMPILE: AGENTS_COMPILE,
    AgentTask.QUERY: AGENTS_QUERY,
    AgentTask.CAPTURE: AGENTS_CAPTURE,
    AgentTask.ORGANIZE: AGENTS_ORGANIZE,
    AgentTask.GENERIC: AGENTS_GENERIC,
}

logger = get_logger(__name__)


class AgentRegistry:
    """Loads and manages agent definitions from built-in + vault sources.

    Usage::

        registry = AgentRegistry(base_dir)
        agent = registry.active_for(AgentTask.COMPILE, config)
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = Path(base_dir) if base_dir else None
        self._cache: dict[str, AgentDefinition] = {}

    @property
    def vault_agents_dir(self) -> Path | None:
        """``.kompend/agents/`` directory, or *None* if no base_dir set."""
        return self._base_dir / ".kompend" / "agents" if self._base_dir else None

    def discover(self) -> dict[str, AgentDefinition]:
        """Load all agents from built-in + vault sources.

        Returns:
            Dict mapping agent name → definition. Vault overrides built-in
            on name collision.
        """
        if self._cache:
            return self._cache

        agents: dict[str, AgentDefinition] = {}

        # 1. Built-in agents
        for toml_file in sorted(_BUILTIN_DIR.glob("*.toml")):
            try:
                definition = AgentDefinition.from_toml(toml_file)
                agents[definition.name] = definition
            except Exception:
                logger.exception("Failed to load built-in agent: %s", toml_file)

        # 2. Vault agents (override built-in on name collision)
        if self.vault_agents_dir and self.vault_agents_dir.is_dir():
            for toml_file in sorted(self.vault_agents_dir.glob("*.toml")):
                try:
                    definition = AgentDefinition.from_toml(toml_file)
                    agents[definition.name] = definition
                except Exception:
                    logger.exception("Failed to load vault agent: %s", toml_file)

        self._cache = agents
        return agents

    def get(self, name: str) -> AgentDefinition | None:
        """Return a specific agent by name, or *None* if not found."""
        return self.discover().get(name)

    def list_by_task(self) -> dict[AgentTask, list[AgentDefinition]]:
        """Group all agents by task category."""
        result: dict[AgentTask, list[AgentDefinition]] = {task: [] for task in AgentTask}
        for agent in self.discover().values():
            result[agent.task].append(agent)
        for agents in result.values():
            agents.sort(key=lambda a: a.name)
        return result

    def active_for(self, task: AgentTask, config: dict[str, Any]) -> AgentDefinition:
        """Resolve the active agent for *task* from config, with fallback.

        Falls back to the generic agent if:
        - No agent name is configured for *task*
        - The configured agent name is not found

        Args:
            task: The task category.
            config: Resolved config dict (from :func:`~kompend.core.config.resolve_config`)
                    or a mapping that supports ``.get(key)``.

        Returns:
            The active :class:`AgentDefinition`. Always returns a valid agent
            (generic as last resort).
        """
        agents = self.discover()
        config_key = _TASK_CONFIG_KEY.get(task, AGENTS_GENERIC)
        agent_name = _config_get(config, config_key)

        if agent_name and agent_name in agents:
            return agents[agent_name]

        # Fall back to generic
        generic = agents.get("generic")
        if generic is None:
            # Last resort: build a minimal generic inline (should never happen
            # because the built-in generic.toml always loads)
            generic = AgentDefinition(
                name="generic",
                description="Fallback generic agent (built-in failed to load)",
                task=AgentTask.GENERIC,
                llm=AgentLLMConfig(),
                prompt=AgentPromptConfig(system="You are a KOMPEND assistant."),
            )
        return generic


def _config_get(config: dict[str, Any], key: str) -> str | None:
    """Read a dotted key from a config dict or config-like object.

    Handles both raw dict and resolved config objects that support ``.get()``.
    """
    get = getattr(config, "get", None)
    if get is not None:
        result = get(key)
        return str(result) if result is not None else None
    return None
