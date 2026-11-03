"""PydanticAI tools for the agentic query agent.

Provides ``build_query_tools()`` returning the toolset the agentic
query agent uses to investigate sub-questions:

- ``search_wiki(question)`` — search the wiki for relevant content.
- ``read_page(slug)`` — read a full wiki page.
- ``get_figures(slug)`` — list the figures a page shows, with captions.
- ``get_tables(slug)`` — read the tables a page shows.
- ``add_finding(sub_question_index, finding)`` — record a finding.

``search_wiki`` / ``read_page`` call the raw retrieval surface on
``QueryAgentDeps.retriever`` — never ``synthesize_results`` — so the agent
cannot re-enter itself. All tools access their data through
``RunContext[QueryAgentDeps]``.
"""

from __future__ import annotations

from pydantic_ai import RunContext, Tool

from kompend.agents.types import QueryAgentDeps
from kompend.core.text.page_media import find_page_figures, find_page_tables
from kompend.llm.format import LlmDataFormat, encode_for_llm
from kompend.types import JSONDict, JSONList


def build_query_tools() -> list[Tool[QueryAgentDeps]]:
    """Return the full toolset for the agentic query agent."""
    return [
        Tool(search_wiki, takes_ctx=True),
        Tool(read_page, takes_ctx=True),
        Tool(get_figures, takes_ctx=True),
        Tool(get_tables, takes_ctx=True),
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
            "excerpt": hit.content,
            "figures": [{"caption": str(f.get("caption", "")), "path": str(f.get("path", ""))} for f in hit.figures],
        }
        payload.append(row)
    return encode_for_llm(payload, LlmDataFormat.JSON)


async def get_figures(ctx: RunContext[QueryAgentDeps], slug: str) -> str:
    """List the figures a wiki page shows, with their captions.

    Use it when a question is about a diagram, chart or picture: the caption
    says what the figure shows, and the path is where the image lives in the
    wiki.
    """
    body = ctx.deps.retriever.read_page(slug)
    if body is None:
        return f"Page not found: {slug}"
    figures = find_page_figures(body)
    if not figures:
        return f"Page shows no figures: {slug}"
    payload: JSONList = [{"caption": caption, "path": path} for caption, path in figures]
    return encode_for_llm(payload, LlmDataFormat.JSON)


async def get_tables(ctx: RunContext[QueryAgentDeps], slug: str) -> str:
    """Read the tables a wiki page shows, as markdown.

    Use it when a question is about numbers, parameters or comparisons that a
    page keeps in a table.
    """
    body = ctx.deps.retriever.read_page(slug)
    if body is None:
        return f"Page not found: {slug}"
    tables = find_page_tables(body)
    if not tables:
        return f"Page shows no tables: {slug}"
    payload: JSONList = [{"table": table} for table in tables]
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
