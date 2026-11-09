"""PydanticAI Tool factories for agentic generation.

Provides ``build_gen_tools()`` which returns the tools an agentic
generation agent needs:

- ``read_segment()`` — returns the full source segment content on demand.
- ``search_sources(query)`` — names the sources of this run that mention a
  query, with the offset of the match inside each one.
- ``read_source_window(stem, offset)`` — serves any source in windows that
  name the next offset, so a source named by a search is read in full.
- ``fetch_concepts(limit)`` — returns the next batch of unassigned concepts
  for this caller from the shared :class:`ConceptBroker`.
- ``list_known_titles(offset)`` — returns the next window of canonical page
  titles the model may link to; the list grows with the corpus, so it is
  served in windows rather than cut down to the head.

All tools access their data through ``RunContext[ToolGenerationDeps]`` so
no closure-captured state is needed.
"""

from __future__ import annotations

import asyncio
from typing import Any, Final

from pydantic_ai import RunContext, Tool

from kompend.agents.types import ToolGenerationDeps
from kompend.core.config._configs import LlmDataFormat
from kompend.llm.format import encode_for_llm

#: Search hits one ``search_sources()`` reply carries. A hit is a preview plus
#: an offset, so the cap bounds the reply, not the reachable text.
MAX_SEARCH_HITS: int = 20
#: Stems named in an "unknown source" reply before the count replaces the list.
_LISTED_STEMS: Final[int] = 20


def build_gen_tools() -> list[Tool[ToolGenerationDeps]]:
    """Return the list of PydanticAI Tool objects for agentic generation."""
    return [
        Tool(read_segment),
        Tool(search_sources),
        Tool(read_source_window),
        Tool(fetch_concepts),
        Tool(list_known_titles),
    ]


async def read_segment(ctx: RunContext[ToolGenerationDeps]) -> str:
    """Return the full source segment content.

    Args:
        ctx: Runtime context providing the source content via deps.

    Call this once at the start of your work to load the source content
    you'll generate pages from.
    """
    return ctx.deps.source_content


async def search_sources(
    ctx: RunContext[ToolGenerationDeps],
    query: str,
    k: int = 6,
) -> str:
    """Find the sources of this run that mention *query*.

    Args:
        ctx: Runtime context providing the source index via deps.
        query: Words to look for — names, acronyms, technical terms.
        k: Maximum number of sources to return (default 6, max 20).

    Each hit names a source, the offset where its preview starts, and the
    preview itself. Read on from there with **read_source_window(stem,
    offset)** — the preview is a window into the match, not the whole of it.
    Use this when a concept's page needs material from a source outside your
    segment; the segment content stays the primary source of truth.
    """
    index = ctx.deps.source_index
    if index is None:
        return "Source search is not available in this run — work from the segment content."
    # search() is synchronous and may call an embedding or judge endpoint;
    # hopping to a worker thread keeps that off the event loop driving the
    # generation agent (and lets the judge's sync entry point run legally).
    hits = await asyncio.to_thread(index.search, query, k=max(1, min(k, MAX_SEARCH_HITS)))
    if not hits:
        return f"No source of this run mentions {query!r}."
    lines = [f"{len(hits)} source(s) match {query!r}:"]
    for hit in hits:
        lines.append(f"- {hit.stem} — preview at offset {hit.offset} of {hit.total_chars} characters (score {hit.score:.2f}):\n  {hit.snippet}")
    lines.append("Call read_source_window(stem, offset) to read any of them in full.")
    return "\n".join(lines)


async def read_source_window(
    ctx: RunContext[ToolGenerationDeps],
    stem: str,
    offset: int = 0,
) -> str:
    """Return a window of the source named *stem*.

    Args:
        ctx: Runtime context providing the source index via deps.
        stem: Source label, as named by ``search_sources()``.
        offset: Character offset to start from (the reply names the next one).

    A source is longer than one call may carry, so it is served in windows
    of ``SOURCE_WINDOW_CHARS`` characters and the reply names the offset of
    the next window — read it to get the rest. Nothing is dropped: the size
    of a window shortens one reply, never the source.
    """
    index = ctx.deps.source_index
    if index is None:
        return "Source reading is not available in this run — work from the segment content."
    window, next_offset = index.read_window(stem, max(0, offset))
    if not window:
        known = index.stems()
        listing = ", ".join(known) if len(known) <= _LISTED_STEMS else f"{len(known)} sources — call search_sources(query) to find one"
        return f"Unknown source {stem!r} (or the offset is past its end). Sources: {listing}"
    total = index.total_chars(stem)
    if next_offset >= total:
        return window
    return f'{window}\n\n[end of window — {next_offset} of {total} characters read; call read_source_window(stem="{stem}", offset={next_offset}) for the rest]'


async def list_known_titles(
    ctx: RunContext[ToolGenerationDeps],
    offset: int = 0,
) -> str:
    """Return a window of the canonical page titles you may link to.

    Args:
        ctx: Runtime context providing the title list via deps.
        offset: Index of the first title to return (the reply names the
            offset of the next window).

    The list grows with the corpus — every concept name plus the previous
    compile's learned aliases — so it is served in windows of
    ``deps.known_titles_page_chars`` characters. Nothing is dropped: the
    prompt inlines only the head, and this tool reaches the rest.
    """
    titles = ctx.deps.known_titles
    total = len(titles)
    if not total:
        return "No canonical page titles are known for this compile."
    start = max(0, min(offset, total))
    if start >= total:
        return f"[end of known titles — all {total} titles have been listed]"
    limit = ctx.deps.known_titles_page_chars
    window: list[str] = []
    used = 0
    for title in titles[start:]:
        line = f"- {title}\n"
        if window and limit > 0 and used + len(line) > limit:
            break
        window.append(line)
        used += len(line)
    end = start + len(window)
    body = f"Canonical page titles ({start + 1}-{end} of {total}):\n{''.join(window)}"
    if end >= total:
        return body
    return f"{body}[end of window — call list_known_titles(offset={end}) for the next {total - end}]"


async def fetch_concepts(
    ctx: RunContext[ToolGenerationDeps],
    limit: int = 10,
) -> str:
    """Fetch the next batch of unassigned concepts for this agent.

    Args:
        ctx: Runtime context providing the concept broker via deps.
        limit: Maximum number of concepts to return (default 10, max 20).

    Returns:
        A JSON-encoded list of concept dicts, or ``[]`` when all
        concepts for this segment have been exhausted.
    """
    broker = ctx.deps.concept_broker
    if broker is None:
        return "[]"
    batch: list[dict[str, Any]] = broker.fetch(ctx.deps.source_id, limit=max(limit, 1))
    if not batch:
        return "[]"
    return encode_for_llm(batch, LlmDataFormat.JSON)
