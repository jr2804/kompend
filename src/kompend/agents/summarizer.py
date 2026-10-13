"""SummarizerAgent — PydanticAI agent for chunk summarization."""

from __future__ import annotations

from typing import ClassVar

from kompend.agents.base import KompendAgentBase
from kompend.agents.deps import PipelineAgentDeps


class SummarizerAgent(KompendAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent for generating chunk summaries."""

    template_name: ClassVar[str] = "node_summary.md.j2"
    system_template_name: ClassVar[str] = "node_summary.md.j2"
    template_prefix: ClassVar[str] = "chunking/"
