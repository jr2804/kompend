"""AgenticContextChunkAgent — PydanticAI agent for chunked contextual enrichment."""

from __future__ import annotations

from typing import ClassVar

from knox.agents.base import KnoxAgentBase
from knox.agents.types import ContextChunkDeps


class AgenticContextChunkAgent(KnoxAgentBase[str, ContextChunkDeps]):
    """Agentic agent that enriches chunks with situating context.

    Iterates over chunk nodes, generating a context paragraph for each
    that situates it within the document. Context strings are collected
    in ``deps.results`` keyed by chunk id.
    """

    template_name: ClassVar[str] = "contextual_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "contextual_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "chunking/"
    use_delta_retry: bool = True

    def render_user_prompt(self, deps: ContextChunkDeps) -> str:  # noqa: PLR6301
        """Return a task description with the chunk to situate."""
        chunk = deps.current_chunk
        if not chunk:
            return "No chunk to situate."
        return (
            f"Generate situating context for this chunk:\n"
            f"Document: {deps.doc_title}\n"
            f"Summary: {deps.doc_summary}\n"
            f"Heading: {chunk.get('heading_chain', [])} \n"
            f"Text: {chunk.get('text', '')[:500]}"
        )
