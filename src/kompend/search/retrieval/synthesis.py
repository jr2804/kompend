"""Synthesis: format retrieved pages into context and let an agent answer.

One entry point: :func:`synthesize_results` pre-retrieves and formats the
results as the first-turn context of the unified query session. Passing a
``retriever`` registers the wiki tools, so the agent can search and read
pages itself when the provided context is insufficient.
"""

from __future__ import annotations

import re

from kompend.agents.failure import record_unit_failure
from kompend.agents.query import QueryAgent, make_query_agent
from kompend.agents.types import QueryAgentDeps, QuerySearchProtocol
from kompend.core.config import ReasoningEffort, RlmConfig
from kompend.core.config._configs import LLMConfig
from kompend.core.vocab import Verbosity
from kompend.search.retrieval.results import Citation, SearchResult, SynthesisResult

_SOURCE_MARKER_RE = re.compile(r"\[sources/[^\]]+\]|\[\^[^\]]+\]")


def synthesize_results(
    query: str,
    results: list[SearchResult],
    model: str = "",
    *,
    verbosity: str = Verbosity.STANDARD,
    skills: dict[str, str] | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    rlm: RlmConfig | None = None,
    query_agent: QueryAgent | None = None,
    retriever: QuerySearchProtocol | None = None,
    context_window: int | None = None,
    llm: LLMConfig | None = None,
) -> SynthesisResult:
    """Format search results into context and call the query session.

    Takes already-retrieved ``SearchResult`` objects, formats them as
    numbered context lines, and passes them to :class:`QueryAgent` as the
    first-turn context. With a ``retriever``, the wiki tools are
    registered and the model may search and read pages itself when the
    context is insufficient. The ``rlm`` kwarg enables RLM code-
    exploration over the synthesized context (``compile.llm.rlm`` config).
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
            agent = make_query_agent(
                model,
                retriever=retriever,
                skills=skills or {},
                reasoning_effort=reasoning_effort,
                rlm=rlm,
                context_window=context_window,
                llm=llm,
            )
        deps = QueryAgentDeps(
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
