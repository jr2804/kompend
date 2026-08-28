"""PydanticAI tools for the agentic write-page synthesis agent.

Provides ``build_write_tools()`` returning the toolset the
agentic write agent uses to build a wiki page from sources:

- ``read_source(stem)`` — load the full text of a source by its stem.
- ``read_related()`` — return the list of related page titles.
- ``save_page(content)`` — submit the final markdown page content.

All tools access their data through ``RunContext[WriteDeps]``.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext

from knox.agents.types import WriteDeps
from knox.llm.format import LlmDataFormat, encode_for_llm


def build_write_tools() -> list[Any]:
    """Return the full toolset for the agentic write agent."""
    return [read_source, list_sources, read_related, save_page]


async def read_source(ctx: RunContext[WriteDeps], stem: str) -> str:
    """Return the full text of the source identified by *stem*."""
    for src_stem, text in ctx.deps.source_texts:
        if src_stem == stem:
            return text
    return f"Unknown source: {stem}"


async def list_sources(ctx: RunContext[WriteDeps]) -> str:
    """Return the list of available source stems."""
    stems = [s for s, _ in ctx.deps.source_texts]
    return encode_for_llm(stems, LlmDataFormat.JSON)


async def read_related(ctx: RunContext[WriteDeps]) -> str:
    """Return the list of related wiki page titles for cross-linking."""
    return encode_for_llm(ctx.deps.related_titles, LlmDataFormat.JSON)


async def save_page(ctx: RunContext[WriteDeps], content: str) -> str:
    """Submit the final markdown page content."""
    ctx.deps.page_content = content
    return "Page saved."
