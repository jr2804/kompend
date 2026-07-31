"""ContextChunkAgent — PydanticAI agent for contextual chunk enrichment."""

from __future__ import annotations
from typing import ClassVar

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps


class ContextChunkAgent(KnoxAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent for generating situating context for a chunk."""

    template_name: ClassVar[str] = "contextual_context.md.j2"
    system_template_name: ClassVar[str] = "contextual_context.md.j2"
    template_prefix: ClassVar[str] = "chunking/"
    temperature: float | None = 0.0
