"""PydanticAI tools for the agentic query agent.

Provides ``build_query_tools()`` returning the toolset the agentic
query agent uses to investigate sub-questions:

- ``search_wiki(question)`` — search the wiki for relevant content.
- ``read_page(slug)`` — read a full wiki page.
- ``add_finding(sub_question_index, finding)`` — record a finding.

All tools access their data through ``RunContext[QueryAgentDeps]``.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext, Tool

from knox.agents.types import QueryAgentDeps


def build_query_tools() -> list[Any]:
    """Return the full toolset for the agentic query agent."""
    return [
        Tool(search_wiki, takes_ctx=True),
        Tool(read_page, takes_ctx=True),
        Tool(add_finding, takes_ctx=True),
    ]


async def search_wiki(ctx: RunContext[QueryAgentDeps], question: str, top_k: int = 5) -> str:
    """Search the wiki for content related to a question."""
    # Placeholder: actual search would dispatch through retriever
    return f"Wiki search for: '{question}' (top {top_k})\n(wiki index lookup — placeholder until retriever is wired)"


async def read_page(ctx: RunContext[QueryAgentDeps], slug: str) -> str:
    """Read a full wiki page by its slug."""
    return f"Page content for '{slug}' (placeholder until page reader is wired)."


async def add_finding(ctx: RunContext[QueryAgentDeps], sub_question_index: int, finding: str) -> str:
    """Record a finding for a sub-question."""
    ctx.deps.results[sub_question_index] = finding.strip()
    return f"Finding recorded for sub-question {sub_question_index}."
