"""PydanticAI Tool factories for agentic generation.

Provides ``build_gen_tools()`` which returns the two tools an agentic
generation agent needs:

- ``read_segment()`` — returns the full source segment content on demand.
- ``fetch_concepts(limit)`` — returns the next batch of unassigned concepts
  for this caller from the shared :class:`ConceptBroker`.

Both tools access their data through ``RunContext[ToolGenerationDeps]`` so
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
    ]


async def read_segment(ctx: RunContext[ToolGenerationDeps]) -> str:
    """Return the full source segment content.

    Args:
        ctx: Runtime context providing the source content via deps.

    Call this once at the start of your work to load the source content
    you'll generate pages from.
    """
    return ctx.deps.source_content


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
