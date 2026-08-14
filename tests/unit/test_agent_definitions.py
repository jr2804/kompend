"""Unit tests for TOML-defined conversational agent definitions and registry."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from knox.acp.definitions import (
    AgentDefinition,
    AgentLLMConfig,
    AgentPromptConfig,
    AgentTask,
)
from knox.acp.registry import AgentRegistry

# ── AgentDefinition ────────────────────────────────────────────────────


class TestAgentDefinition:
    """Pydantic model for TOML-defined agents."""

    def test_minimal_valid(self) -> None:
        """Minimal agent with only required fields."""
        agent = AgentDefinition(
            name="test",
            description="Test agent",
            task=AgentTask.GENERIC,
            prompt=AgentPromptConfig(system="You are a test agent."),
        )
        assert agent.name == "test"
        assert agent.task == AgentTask.GENERIC
        assert agent.llm.model is None
        assert agent.llm.temperature == 0.3
        assert agent.tools.enabled is None  # all tools
        assert agent.behavior.max_turns == 10

    def test_from_toml(self, tmp_path: Path) -> None:
        """Load from a TOML file."""
        toml_content = textwrap.dedent("""\
            name = "my-agent"
            description = "Custom agent"
            task = "compile"

            [llm]
            model = "ollama-cloud/deepseek-v4-flash"
            temperature = 0.1

            [prompt]
            system = "You are a compilation assistant."

            [tools]
            enabled = ["compile_wiki", "wiki_status"]

            [behavior]
            max_turns = 5
            stream_progress = false
            """)
        toml_file = tmp_path / "my-agent.toml"
        toml_file.write_text(toml_content, encoding="utf-8")

        agent = AgentDefinition.from_toml(toml_file)
        assert agent.name == "my-agent"
        assert agent.task == AgentTask.COMPILE
        assert agent.llm.model == "ollama-cloud/deepseek-v4-flash"
        assert agent.llm.temperature == 0.1
        assert agent.tools.enabled == ["compile_wiki", "wiki_status"]
        assert agent.behavior.max_turns == 5
        assert agent.behavior.stream_progress is False

    def test_from_toml_missing_model_uses_none(self, tmp_path: Path) -> None:
        """Omitting [llm].model means None (vault default)."""
        toml_content = textwrap.dedent("""\
            name = "no-model"
            description = "Agent without explicit model"
            task = "query"

            [prompt]
            system = "Answer questions."
            """)
        toml_file = tmp_path / "no-model.toml"
        toml_file.write_text(toml_content, encoding="utf-8")

        agent = AgentDefinition.from_toml(toml_file)
        assert agent.llm.model is None

    def test_from_toml_missing_file_raises(self, tmp_path: Path) -> None:
        """Missing TOML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            AgentDefinition.from_toml(tmp_path / "nonexistent.toml")

    def test_invalid_temperature_rejected(self) -> None:
        """Temperature outside [0, 2] is rejected."""
        with pytest.raises(ValidationError):
            AgentDefinition(
                name="bad",
                description="Bad agent",
                task=AgentTask.GENERIC,
                llm=AgentLLMConfig(temperature=5.0),
                prompt=AgentPromptConfig(system="x"),
            )

    def test_extra_keys_rejected(self, tmp_path: Path) -> None:
        """Unknown top-level keys are rejected (strict schema)."""
        toml_content = textwrap.dedent("""\
            name = "extra"
            description = "Agent with extra keys"
            task = "generic"
            unknown_key = true

            [prompt]
            system = "x"
            """)
        toml_file = tmp_path / "extra.toml"
        toml_file.write_text(toml_content, encoding="utf-8")

        with pytest.raises(ValidationError):
            AgentDefinition.from_toml(toml_file)

    def test_all_task_values(self) -> None:
        """Every AgentTask value produces a valid agent."""
        for task in AgentTask:
            agent = AgentDefinition(
                name=f"agent-{task.value}",
                description=f"Agent for {task.value}",
                task=task,
                prompt=AgentPromptConfig(system="x"),
            )
            assert agent.task == task


# ── AgentRegistry ──────────────────────────────────────────────────────


class TestAgentRegistry:
    """Discovery and selection of agents."""

    def test_builtin_generic_loaded(self) -> None:
        """The built-in generic.toml loads without a base_dir."""
        registry = AgentRegistry()
        agents = registry.discover()
        assert "generic" in agents
        generic = agents["generic"]
        assert generic.task == AgentTask.GENERIC
        assert generic.tools.enabled is None  # all tools

    def test_vault_agent_overrides_builtin(self, tmp_path: Path) -> None:
        """Vault agent with same name as built-in overrides it."""
        vault_agents = tmp_path / ".knox" / "agents"
        vault_agents.mkdir(parents=True)
        # Override generic with a custom prompt
        (vault_agents / "generic.toml").write_text(
            textwrap.dedent("""\
                name = "generic"
                description = "Custom override"
                task = "generic"

                [prompt]
                system = "You are a CUSTOM agent."
                """),
            encoding="utf-8",
        )

        registry = AgentRegistry(base_dir=tmp_path)
        agents = registry.discover()
        assert agents["generic"].description == "Custom override"

    def test_vault_agent_added(self, tmp_path: Path) -> None:
        """Vault-only agent is discovered alongside built-in."""
        vault_agents = tmp_path / ".knox" / "agents"
        vault_agents.mkdir(parents=True)
        (vault_agents / "researcher.toml").write_text(
            textwrap.dedent("""\
                name = "researcher"
                description = "Custom research agent"
                task = "query"

                [prompt]
                system = "You are a research assistant."

                [tools]
                enabled = ["query_wiki"]
                """),
            encoding="utf-8",
        )

        registry = AgentRegistry(base_dir=tmp_path)
        agents = registry.discover()
        assert "generic" in agents
        assert "researcher" in agents
        assert agents["researcher"].task == AgentTask.QUERY

    def test_active_for_falls_back_to_generic(self) -> None:
        """No agent configured for a task → generic is returned."""
        registry = AgentRegistry()
        active = registry.active_for(AgentTask.COMPILE, {})
        assert active.name == "generic"

    def test_active_for_resolves_config(self, tmp_path: Path) -> None:
        """Config-specified agent is resolved."""
        vault_agents = tmp_path / ".knox" / "agents"
        vault_agents.mkdir(parents=True)
        (vault_agents / "my-compiler.toml").write_text(
            textwrap.dedent("""\
                name = "my-compiler"
                description = "Custom compiler"
                task = "compile"

                [prompt]
                system = "Compile things."
                """),
            encoding="utf-8",
        )

        registry = AgentRegistry(base_dir=tmp_path)
        config = {"agents.compile": "my-compiler"}
        active = registry.active_for(AgentTask.COMPILE, config)
        assert active.name == "my-compiler"

    def test_active_for_invalid_name_falls_back(self, tmp_path: Path) -> None:
        """Config points to non-existent agent → generic fallback."""
        registry = AgentRegistry(base_dir=tmp_path)
        config = {"agents.compile": "nonexistent-agent"}
        active = registry.active_for(AgentTask.COMPILE, config)
        assert active.name == "generic"

    def test_list_by_task_groups_correctly(self, tmp_path: Path) -> None:
        """Agents are grouped by task."""
        vault_agents = tmp_path / ".knox" / "agents"
        vault_agents.mkdir(parents=True)
        for task in ("compile", "query"):
            (vault_agents / f"{task}-1.toml").write_text(
                f'name = "{task}-1"\ndescription = "x"\ntask = "{task}"\n\n[prompt]\nsystem = "x"\n',
                encoding="utf-8",
            )

        registry = AgentRegistry(base_dir=tmp_path)
        grouped = registry.list_by_task()
        assert len(grouped[AgentTask.COMPILE]) >= 1
        assert len(grouped[AgentTask.QUERY]) >= 1
        assert len(grouped[AgentTask.GENERIC]) >= 1  # built-in

    def test_cache_is_reused(self) -> None:
        """discover() caches; second call returns same dict object."""
        registry = AgentRegistry()
        first = registry.discover()
        second = registry.discover()
        assert first is second
