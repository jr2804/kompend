"""AgenticQueryAgent — multi-turn query synthesis with tools.

One session covers a complex question: the agent decomposes it
into sub-questions, searches/wiki-reads via tools, and synthesizes
the final answer.
"""

from __future__ import annotations

from typing import ClassVar

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import UNSET_RETRIES, UNSET_TIMEOUT, KompendAgentBase, resolve_retry_budget
from kompend.agents.types import QueryAgentDeps
from kompend.compilation.query_tools import build_query_tools
from kompend.core.config._configs import LLMConfig, RlmConfig


class AgenticQueryAgent(KompendAgentBase[str, QueryAgentDeps]):
    """PydanticAI agent that synthesizes answers via tools."""

    template_name: ClassVar[str] = "query_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "query_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "search/"


def make_agentic_query_agent(
    model_string: str,
    *,
    timeout: float = UNSET_TIMEOUT,
    skills: dict[str, str] | None = None,
    retries: int = UNSET_RETRIES,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
    llm: LLMConfig | None = None,
) -> AgenticQueryAgent:
    """Build an :class:`AgenticQueryAgent` with tools + harness capabilities.

    Registers the ``search_wiki()`` / ``read_page()`` / ``add_finding()`` tools
    the system prompt instructs the model to call, and injects the standard
    agentic capabilities via :class:`AgenticHarness`.
    """
    agent_retries = resolve_retry_budget(llm, retries=retries, output_retries=output_retries)
    harness = AgenticHarness(context_window=context_window, memory_agent_name="query")
    return AgenticQueryAgent(
        model_string=model_string,
        timeout=timeout,
        skills=skills or {},
        retries=agent_retries,
        context_window=context_window,
        rlm=rlm,
        llm=llm,
        tools=build_query_tools(),
        extra_capabilities=harness.build_capabilities(),
    )
