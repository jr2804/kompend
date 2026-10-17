"""ContextChunkAgent — PydanticAI agent for contextual chunk enrichment."""

from __future__ import annotations

from typing import ClassVar, override

from kompend.agents.base import KompendAgentBase
from kompend.agents.types import PipelineAgentDeps


class ContextChunkAgent(KompendAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent for generating situating context for a chunk."""

    # The caller renders the prompt: ``chunking.contextual`` fills
    # ``chunking/contextual_context.md.j2`` with the document title, summary,
    # heading path and chunk preview, then passes the finished text as
    # ``PipelineAgentDeps.text``. So this agent renders no template of its own -
    # naming one here made the base render it with no variables at all, which
    # raised on every call and was swallowed by the caller's ``debug`` log,
    # leaving enrichment permanently on its structural fallback.
    template_name: ClassVar[str] = ""
    system_template_name: ClassVar[str] = ""
    template_prefix: ClassVar[str] = "chunking/"
    temperature: float | None = 0.0

    @override
    def render_user_prompt(self, deps: PipelineAgentDeps) -> str:
        """The caller supplies the finished prompt; send it as-is."""
        return deps.text
