"""Pydantic models for TOML-defined conversational agents.

Each agent is a single TOML file with a system prompt, tool list, and
optional model/behavior configuration. Loaded by :class:`AgentRegistry`.

TOML schema::

    name = "wiki-compiler"
    description = "Compiles documents into a knowledge wiki"
    task = "compile"  # compile | query | capture | organize | generic

    [llm]
    model = null  # null = vault default
    temperature = 0.2
    max_tokens = 4096

    [prompt]
    system = "You are a KNOX compilation assistant..."

    [tools]
    enabled = ["compile_wiki", "wiki_status"]  # null/omitted = all tools

    [behavior]
    max_turns = 10
    stream_progress = true
"""

from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentTask(StrEnum):
    """User-facing task categories for agent selection.

    Distinct from :class:`~knox.llm.roles.LLMRole` (pipeline-internal roles).
    Tasks map to the kind of operation the user wants to perform, not to
    individual LLM calls within the pipeline.
    """

    COMPILE = "compile"
    QUERY = "query"
    CAPTURE = "capture"
    ORGANIZE = "organize"
    GENERIC = "generic"


class AgentLLMConfig(BaseModel):
    """Per-agent LLM model and generation parameters."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(default=None, description="Model identifier. None = vault default.")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)


class AgentPromptConfig(BaseModel):
    """System prompt for the agent. Taken as-is from TOML."""

    model_config = ConfigDict(extra="forbid")

    system: str = Field(description="System prompt text. Written as-is; default language is English.")


class AgentToolsConfig(BaseModel):
    """Tool access configuration for the agent."""

    model_config = ConfigDict(extra="forbid")

    enabled: list[str] | None = Field(
        default=None,
        description="Tool names the agent may call. None = all available tools.",
    )


class AgentBehaviorConfig(BaseModel):
    """Behavioral limits for the conversational loop."""

    model_config = ConfigDict(extra="forbid")

    max_turns: int = Field(default=10, ge=1, description="Maximum LLM round-trips per prompt.")
    stream_progress: bool = Field(default=True, description="Forward tool progress as session/update chunks.")


class AgentDefinition(BaseModel):
    """A single TOML-defined conversational agent.

    Loaded from a TOML file via :meth:`from_toml` or constructed directly.
    Used by :class:`~knox.acp.conversational.ConversationalAgent` and
    :class:`~knox.acp.registry.AgentRegistry`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique agent name (matches filename without .toml).")
    description: str = Field(description="Human-readable description shown in `knox agents list`.")
    task: AgentTask = Field(description="Task category for selection.")
    llm: AgentLLMConfig = Field(default_factory=AgentLLMConfig)
    prompt: AgentPromptConfig
    tools: AgentToolsConfig = Field(default_factory=AgentToolsConfig)
    behavior: AgentBehaviorConfig = Field(default_factory=AgentBehaviorConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> AgentDefinition:
        """Load an agent definition from a TOML file.

        Args:
            path: Path to the TOML file.

        Returns:
            Parsed and validated :class:`AgentDefinition`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValidationError: If the TOML content doesn't match the schema.
        """
        path = Path(path)
        with path.open("rb") as f:
            data: dict[str, Any] = tomllib.load(f)
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentDefinition:
        """Construct from a pre-parsed dict (e.g. from config.toml inline agent)."""
        return cls.model_validate(data)
