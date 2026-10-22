"""PydanticAI tools for the agentic write-page synthesis agent.

Provides ``build_write_tools()`` returning the toolset the
agentic write agent uses to build a wiki page from sources:

- ``read_source(stem, offset=0)`` — load a window of a source's text; the
  reply names the offset of the next window, so the agent can page through
  a long source without losing any of it.
- ``read_related()`` — return the list of related page titles.
- ``save_page(content)`` — submit the final markdown page content.

All tools access their data through ``RunContext[WriteDeps]``.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext

from kompend.agents.types import WriteDeps
from kompend.llm.format import LlmDataFormat, encode_for_llm


def build_write_tools() -> list[Any]:
    """Return the full toolset for the agentic write agent."""
    return [read_source, list_sources, read_related, save_page]


async def read_source(ctx: RunContext[WriteDeps], stem: str, offset: int = 0) -> str:
    """Return a window of the source text identified by *stem*.

    A source is longer than one call may carry, so the text is served in
    windows of ``deps.source_char_max`` characters. The reply names the
    offset of the next window — read it to get the rest. Nothing is
    dropped: the budget sizes a window, it does not shorten the source.
    """
    for src_stem, text in ctx.deps.source_texts:
        if src_stem != stem:
            continue
        limit = ctx.deps.source_char_max
        if limit <= 0 or (offset <= 0 and len(text) <= limit):
            return text
        if offset >= len(text):
            return ""
        window = text[offset : offset + limit]
        next_offset = offset + len(window)
        if next_offset >= len(text):
            return window
        return f'{window}\n\n[end of window — {next_offset} of {len(text)} characters read; call read_source(stem="{stem}", offset={next_offset}) for the rest]'
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
