"""AgenticContextChunkAgent — PydanticAI agent for chunked contextual enrichment."""

from __future__ import annotations

from typing import ClassVar

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import UNSET_RETRIES, UNSET_TIMEOUT, KompendAgentBase, resolve_retry_budget
from kompend.agents.types import ContextChunkDeps
from kompend.compilation.context_chunk_tools import CONTEXT_CHUNK_TEXT_PREVIEW_CHARS, build_context_chunk_tools
from kompend.core.config._configs import BudgetConfig, LLMConfig, RlmConfig


class AgenticContextChunkAgent(KompendAgentBase[str, ContextChunkDeps]):
    """Agentic agent that enriches chunks with situating context.

    Iterates over chunk nodes, generating a context paragraph for each
    that situates it within the document. Context strings are collected
    in ``deps.results`` keyed by chunk id.
    """

    template_name: ClassVar[str] = "contextual_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "contextual_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "chunking/"
    supports_delta_retry: ClassVar[bool] = True

    def render_user_prompt(self, deps: ContextChunkDeps) -> str:  # noqa: PLR6301
        """Return a task description with the chunk to situate."""
        chunk = deps.current_chunk
        if not chunk:
            return "No chunk to situate."
        heading = " > ".join(chunk["heading_chain"]) or "document root"
        return (
            f"Generate situating context for this chunk:\n"
            f"Document: {deps.doc_title}\n"
            f"Summary: {deps.doc_summary}\n"
            f"Heading: {heading}\n"
            f"Text: {chunk['text'][:CONTEXT_CHUNK_TEXT_PREVIEW_CHARS]}"
        )


def make_agentic_context_chunk_agent(
    model_string: str,
    *,
    timeout: float = UNSET_TIMEOUT,
    skills: dict[str, str] | None = None,
    retries: int = UNSET_RETRIES,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
    llm: LLMConfig | None = None,
    budget: BudgetConfig | None = None,
) -> AgenticContextChunkAgent:
    """Build an :class:`AgenticContextChunkAgent` with tools + harness capabilities.

    Registers the ``next_chunk()`` / ``situate_chunk()`` tools the system prompt
    instructs the model to call, and injects the standard agentic capabilities
    via :class:`AgenticHarness`.
    """
    agent_retries = resolve_retry_budget(llm, retries=retries, output_retries=output_retries)
    harness = AgenticHarness(context_window=context_window, memory_agent_name="context_chunk")
    return AgenticContextChunkAgent(
        model_string=model_string,
        timeout=timeout,
        skills=skills or {},
        retries=agent_retries,
        context_window=context_window,
        rlm=rlm,
        llm=llm,
        budget=budget,
        tools=build_context_chunk_tools(),
        extra_capabilities=harness.build_capabilities(),
    )
