r"""QueryAgent — the unified wiki query session.

One session-shaped agent serves every query: the first turn carries the
pre-retrieved, deps-rendered context; the wiki tools
(``search_wiki`` / ``read_page`` / ``get_figures`` / ``get_tables`` /
``add_finding``) are registered whenever a retrieval surface is present,
and the model decides whether the provided context suffices or it should
investigate further. A zero-tool session IS one request — the old
single-shot synthesis is the degenerate case, not a sibling path.

The agent returns **free text** — the answer string.
PydanticAI's ``output_type=str`` handles this natively.

Usage::

    from kompend.agents.query import QueryAgent, make_query_agent
    from kompend.agents.types import QueryAgentDeps

    agent = make_query_agent("ollama/gpt-oss:20b", retriever=retriever)
    answer: str = agent.run(
        QueryAgentDeps(
            question="What is VAD?",
            context="[1] Voice Activity Detection\\n    ...",
            source_id="query-1",
        )
    )
"""

from __future__ import annotations

from typing import Any, ClassVar, Final

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import (
    RLM_CONTEXT_PLACEHOLDER,
    UNSET_RETRIES,
    UNSET_TIMEOUT,
    AgentSpec,
    KompendAgentBase,
    resolve_retry_budget,
)
from kompend.agents.types import QueryAgentDeps, QuerySearchProtocol
from kompend.compilation.query_tools import build_query_tools
from kompend.core.config import ReasoningEffort
from kompend.core.config._configs import LLMConfig, RlmConfig


def _query_user_prompt(agent: Any, deps: QueryAgentDeps) -> str:
    """The user prompt is the raw question — no template needed."""
    return deps.question


def _query_system_prompt(agent: Any, deps: QueryAgentDeps) -> str:
    """Render the session template with the upfront context + skills."""
    ctx = deps.context
    if agent.rlm:
        ctx = RLM_CONTEXT_PLACEHOLDER
    prompt = agent._template_loader.render(  # noqa: SLF001
        f"{agent.template_prefix}{agent.system_template_name}",
        context=ctx,
        verbosity=deps.verbosity,
        max_sub_questions=deps.max_sub_questions,
    )
    if agent.skills:
        skill_blocks = [f"## Skill: {name}\n\n{instructions}" for name, instructions in agent.skills.items()]
        prompt += "\n\n" + "\n\n---\n\n".join(skill_blocks)
    return prompt


SPEC: Final[AgentSpec] = AgentSpec(
    name="query",
    result_type=str,
    deps_type=QueryAgentDeps,
    template_name="",
    system_template_name="query_session_system.md.j2",
    template_prefix="query/",
    user_prompt=_query_user_prompt,
    system_prompt=_query_system_prompt,
)


class QueryAgent(KompendAgentBase[str, QueryAgentDeps]):
    """PydanticAI agent for the unified wiki query session.

    The system prompt comes from ``query/query_session_system.md.j2``,
    which folds the synthesis instructions (prose-only answers, inline
    citations, verbosity profile) and the tool workflow into one prompt:
    answer from the provided context first; search only if insufficient.
    The user prompt is the raw question string. Output is free text.
    """

    spec: ClassVar[AgentSpec | None] = SPEC


def make_query_agent(
    model_string: str,
    *,
    retriever: QuerySearchProtocol | None = None,
    timeout: float = UNSET_TIMEOUT,
    skills: dict[str, str] | None = None,
    retries: int = UNSET_RETRIES,
    output_retries: int | None = None,
    context_window: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    rlm: RlmConfig | None = None,
    rlm_context_field: str = "context",
    llm: LLMConfig | None = None,
) -> QueryAgent:
    """Build a :class:`QueryAgent` with harness capabilities.

    Registers the wiki tools only when a *retriever* is present — without
    a retrieval surface the model answers from the upfront context alone.
    The session always runs on :class:`AgenticHarness` capabilities, so a
    tool-using follow-up gets compaction and step persistence for free.
    """
    agent_retries = resolve_retry_budget(llm, retries=retries, output_retries=output_retries)
    harness = AgenticHarness(context_window=context_window)
    return QueryAgent(
        model_string=model_string,
        timeout=timeout,
        skills=skills or {},
        retries=agent_retries,
        context_window=context_window,
        reasoning_effort=reasoning_effort,
        rlm=rlm,
        rlm_context_field=rlm_context_field,
        llm=llm,
        tools=build_query_tools() if retriever is not None else [],
        extra_capabilities=harness.build_capabilities(),
    )
