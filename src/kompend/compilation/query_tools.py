"""PydanticAI tools for the agentic query agent.

Provides ``build_query_tools()`` returning the toolset the agentic
query agent uses to investigate sub-questions:

- ``search_wiki(question)`` — search the wiki for relevant content.
- ``read_page(slug)`` — read a full wiki page.
- ``add_finding(sub_question_index, finding)`` — record a finding.

``search_wiki`` / ``read_page`` call the raw retrieval surface on
``QueryAgentDeps.retriever`` — never ``synthesize_results`` — so the agent
cannot re-enter itself. All tools access their data through
``RunContext[QueryAgentDeps]``.
"""

from __future__ import annotations

from pydantic_ai import RunContext, Tool

from kompend.agents.types import QueryAgentDeps
from kompend.llm.format import LlmDataFormat, encode_for_llm
from kompend.types import JSONDict, JSONList

_MAX_PAGE_CHARS = 2000


def build_query_tools() -> list[Tool[QueryAgentDeps]]:
    """Return the full toolset for the agentic query agent."""
    return [
        Tool(search_wiki, takes_ctx=True),
        Tool(read_page, takes_ctx=True),
        Tool(add_finding, takes_ctx=True),
    ]


async def search_wiki(ctx: RunContext[QueryAgentDeps], question: str, top_k: int = 5) -> str:
    """Search the wiki for content related to a question."""
    tags_dnf = ctx.deps.tag_groups or None
    value_pairs = [f"{key}={value}" for key, value in ctx.deps.attr_filters if value is not None]
    key_only = [key for key, value in ctx.deps.attr_filters if value is None]
    hits = list(
        ctx.deps.retriever.search(
            question,
            top_k,
            tags_dnf=tags_dnf,
            attributes_all=value_pairs or None,
            attribute_keys=key_only or None,
        )
    )
    if not hits:
        return f"No wiki pages matched: {question}"
    payload: JSONList = []
    for hit in hits:
        row: JSONDict = {
            "slug": hit.article_id,
            "title": hit.title,
            "excerpt": hit.content[:_MAX_PAGE_CHARS],
        }
        payload.append(row)
    return encode_for_llm(payload, LlmDataFormat.JSON)


async def read_page(ctx: RunContext[QueryAgentDeps], slug: str) -> str:
    """Read a full wiki page by its slug."""
    body = ctx.deps.retriever.read_page(slug)
    if body is None:
        return f"Page not found: {slug}"
    return body


async def add_finding(ctx: RunContext[QueryAgentDeps], sub_question_index: int, finding: str) -> str:
    """Record a finding for a sub-question."""
    ctx.deps.results[sub_question_index] = finding.strip()
    return f"Finding recorded for sub-question {sub_question_index}."
