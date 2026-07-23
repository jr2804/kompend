"""SummarizerAgent — PydanticAI agent for chunk summarization."""

from __future__ import annotations

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps


class SummarizerAgent(KnoxAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent for generating chunk summaries."""

    template_name: str = "node_summary.md.j2"
    system_template_name: str = "node_summary.md.j2"
    template_prefix: str = "chunking/"
    result_type: type = str
    deps_type: type[PipelineAgentDeps] = PipelineAgentDeps
