"""SummarizerAgent — PydanticAI agent for chunk summarization."""

from __future__ import annotations

from typing import ClassVar, Final

from kompend.agents.base import AgentSpec, KompendAgentBase
from kompend.agents.deps import PipelineAgentDeps

SPEC: Final[AgentSpec] = AgentSpec(
    name="summarizer",
    result_type=str,
    deps_type=PipelineAgentDeps,
    template_name="node_summary.md.j2",
    system_template_name="node_summary.md.j2",
    template_prefix="chunking/",
)


class SummarizerAgent(KompendAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent for generating chunk summaries."""

    spec: ClassVar[AgentSpec | None] = SPEC
