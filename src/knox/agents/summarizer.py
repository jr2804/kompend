"""SummarizerAgent — PydanticAI agent for chunk summarization."""

from __future__ import annotations

from typing import ClassVar

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps


class SummarizerAgent(KnoxAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent for generating chunk summaries."""

    template_name: ClassVar[str] = "node_summary.md.j2"
    system_template_name: ClassVar[str] = "node_summary.md.j2"
    template_prefix: ClassVar[str] = "chunking/"
