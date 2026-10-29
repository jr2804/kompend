"""PydanticAI Tool factories for agentic generation.

Provides ``build_gen_tools()`` which returns the tools an agentic
generation agent needs:

- ``read_segment()`` — returns the full source segment content on demand.
- ``fetch_concepts(limit)`` — returns the next batch of unassigned concepts
  for this caller from the shared :class:`ConceptBroker`.
- ``list_known_titles(offset)`` — returns the next window of canonical page
  titles the model may link to; the list grows with the corpus, so it is
  served in windows rather than cut down to the head.

All tools access their data through ``RunContext[ToolGenerationDeps]`` so
no closure-captured state is needed.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext, Tool

from kompend.agents.types import ToolGenerationDeps
from kompend.core.config._configs import LlmDataFormat
from kompend.llm.format import encode_for_llm


def build_gen_tools() -> list[Tool[ToolGenerationDeps]]:
    """Return the list of PydanticAI Tool objects for agentic generation."""
    return [
        Tool(read_segment),
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
