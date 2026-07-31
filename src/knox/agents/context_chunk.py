"""ContextChunkAgent — PydanticAI agent for contextual chunk enrichment."""

from __future__ import annotations

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps


class ContextChunkAgent(KnoxAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent for generating situating context for a chunk."""

    template_name: str = "contextual_context.md.j2"
    system_template_name: str = "contextual_context.md.j2"
    template_prefix: str = "chunking/"
    temperature: float | None = 0.0
