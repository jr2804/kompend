"""Synthesis: format retrieved pages into context and let an agent answer.

Two entry points with different shapes, kept separate on purpose:
:func:`synthesize_results` pre-retrieves and formats results for a
single-shot :class:`QueryAgent`, while :func:`run_agentic_query` hands the
agent the raw retrieval surface and lets it search and read pages itself.
"""

from __future__ import annotations

import re

from kompend.agents.base import UNSET_RETRIES, UNSET_TIMEOUT
from kompend.agents.deps import QueryDeps
from kompend.agents.failure import record_unit_failure
from kompend.agents.query import QueryAgent
from kompend.agents.query_agentic import make_agentic_query_agent
from kompend.agents.types import QueryAgentDeps, QuerySearchProtocol
from kompend.core.config import ReasoningEffort, RlmConfig
from kompend.core.config._configs import LLMConfig
from kompend.search.retrieval.results import Citation, SearchResult, SynthesisResult

_SOURCE_MARKER_RE = re.compile(r"\[sources/[^\]]+\]|\[\^[^\]]+\]")


def synthesize_results(
    query: str,
    results: list[SearchResult],
    model: str = "",
    *,
    verbosity: str = "standard",
    skills: dict[str, str] | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    rlm: RlmConfig | None = None,
    query_agent: QueryAgent | None = None,
    llm: LLMConfig | None = None,
) -> SynthesisResult:
    """Format search results into context and call the LLM for an answer.

    This is a standalone function (no retriever needed) — it takes already-
    retrieved ``SearchResult`` objects, formats them as numbered context
    lines, and passes them to :class:`QueryAgent`. The ``rlm`` kwarg enables
    RLM code-exploration over the synthesized context
    (``compile.llm.rlm`` config).
    """
    if not model:
        citations: list[Citation] = []
        for r in results:
            citations.append(
                {
                    "article": r.title,
                    "article_id": r.article_id,
                    "source": r.source_file,
                    "snippet": r.snippet,
                    "score": round(r.score, 3),
                    "figures": r.figures,
                }
            )
        return SynthesisResult(answer="", citations=citations)

    context_lines: list[str] = []
    for i, r in enumerate(results, 1):
        context_lines.append(f"[{i}] {r.title}")
        if r.source_file:
            context_lines.append(f"    Source: {r.source_file}")
        # Full page body when available — the snippet window (~300 chars)
        # is display-only and usually misses the passage that answers the
        # query. Fall back to the snippet for graph-expansion-only results.
        body = r.content.strip() or r.snippet
        # Strip inline source markers ([sources/foo.pdf]) — small models
        # mine them and emit citation arrays instead of prose.
        body = _SOURCE_MARKER_RE.sub("", body)
        context_lines.append(f"    {body}")
        context_lines.append("")

    context_text: str = "\n".join(context_lines)

    try:
        if query_agent is not None:
            agent = query_agent
        else:
            agent = QueryAgent(
                model_string=model,
                skills=skills or {},
                reasoning_effort=reasoning_effort,
                rlm=rlm,
                rlm_context_field="context",
                llm=llm,
            )
        deps = QueryDeps(
            question=query,
            context=context_text,
            source_id="query",
            verbosity=verbosity,
        )
        answer = agent.run(deps)
    except Exception as exc:
        # No fallback answer: an empty string would be returned as if it were
        # the model's answer. Every caller (CLI, MCP) already handles a
        # raised synthesis by showing the raw context instead.
        record_unit_failure(stage="search.synthesis", unit=query, exc=exc)
        raise

    citations_out: list[Citation] = []
    for r in results:
        citations_out.append(
            {
                "article": r.title,
                "article_id": r.article_id,
                "source": r.source_file,
                "snippet": r.snippet,
                "score": round(r.score, 3),
                "figures": r.figures,
            }
        )

    return SynthesisResult(answer=answer, citations=citations_out)


def run_agentic_query(
    question: str,
    *,
    retriever: QuerySearchProtocol,
    model: str,
    max_sub_questions: int = 5,
    timeout: float = UNSET_TIMEOUT,
    retries: int = UNSET_RETRIES,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
    tag_groups: list[list[str]] | None = None,
    attr_filters: list[tuple[str, str | None]] | None = None,
    llm: LLMConfig | None = None,
) -> str:
    """Answer *question* with the agentic query agent against *retriever*.

    Unlike :func:`synthesize_results` — which pre-retrieves and formats
    results for a single-shot :class:`QueryAgent` — this hands the agent the
    question plus the raw retrieval surface and lets it search/read pages via
    tools. Deliberately separate: ``synthesize_results`` builds
    :class:`QueryDeps`, which has no ``retriever``, so an agentic agent run
    there would crash the query tools.
    """
    agent = make_agentic_query_agent(
        model,
        timeout=timeout,
        retries=retries,
        context_window=context_window,
        rlm=rlm,
        llm=llm,
    )
    deps = QueryAgentDeps(
        question=question,
        sub_questions=[],
        retriever=retriever,
        max_sub_questions=max_sub_questions,
        tag_groups=tag_groups or [],
        attr_filters=attr_filters or [],
    )
    return agent.run(deps)
