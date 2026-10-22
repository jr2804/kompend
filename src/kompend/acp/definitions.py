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

    [prompt]
    system = "You are a kompend compilation assistant..."

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

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kompend.logging import get_logger


class AgentTask(StrEnum):
    """User-facing task categories for agent selection.

    Tasks map to the kind of operation the user wants to perform, not to
    individual LLM calls within the pipeline.
    """

    COMPILE = "compile"
    QUERY = "query"
    CAPTURE = "capture"
    ORGANIZE = "organize"
    GENERIC = "generic"


class AgentLLMConfig(BaseModel):
    """Per-agent LLM model and sampling parameters.

    There is no output ceiling: a reply is bounded by the model's context
    window, so a ``max_tokens`` entry is dropped from older definitions with
    a warning instead of failing the agent.
    """

    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(default=None, description="Model identifier. None = vault default.")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_max_tokens(cls, data: object) -> object:
        """Ignore a legacy ``max_tokens`` key — kompend never sends a ceiling."""
        if isinstance(data, dict) and "max_tokens" in data:
            warning = "ACP agent [llm].max_tokens is ignored: a reply is bounded by the model's context window"
            get_logger(__name__).warning(warning)
            return {key: value for key, value in data.items() if key != "max_tokens"}
        return data


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
    Used by :class:`~kompend.acp.conversational.ConversationalAgent` and
    :class:`~kompend.acp.registry.AgentRegistry`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique agent name (matches filename without .toml).")
    description: str = Field(description="Human-readable description shown in `kompend agents list`.")
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
