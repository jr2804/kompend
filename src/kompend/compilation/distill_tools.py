"""PydanticAI tools for the agentic distill agent.

Provides ``build_distill_tools()`` returning the toolset the agentic
distill agent uses to process tag groups:

- ``get_group(tag)`` — return the notes for a tag group.
- ``save_page(tag, page)`` — record a distilled wiki page for a group.

All tools access their data through ``RunContext[DistillAgentDeps]``.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic_ai import RunContext, Tool

from kompend.agents.types import DistillAgentDeps, DistillPage
from kompend.llm.format import encode_payload
from kompend.types import JSONDict


def build_distill_tools() -> list[Any]:
    """Return the full toolset for the agentic distill agent."""
    return [
        Tool(get_group, takes_ctx=True),
        Tool(save_page, takes_ctx=True),
    ]


async def get_group(ctx: RunContext[DistillAgentDeps], tag: str) -> str:
    """Return the notes belonging to a tag group."""
    for g in ctx.deps.groups:
        if g["tag"] == tag:
            ctx.deps.current_group = g
            # ``DistillNote`` is already the JSON shape the model expects; a
            # cast states that, instead of re-typing its keys here.
            notes = cast("list[JSONDict]", g["notes"])
            notes_data = encode_payload(notes, ctx.deps.data_format)
            return f"Group '{tag}' with {len(g['notes'])} notes:\n{notes_data}"
    return f"Group '{tag}' not found. Available: {[g['tag'] for g in ctx.deps.groups]}"


async def save_page(ctx: RunContext[DistillAgentDeps], tag: str, page: dict[str, Any]) -> str:
    """Record a distilled wiki page for a tag group.

    ``page`` is untrusted model output: every key of :class:`DistillPage` is
    optional and the caller re-reads ``title``/``content``/``tags`` defensively.
    """
    ctx.deps.pages[tag] = cast("DistillPage", page)
    ctx.deps.current_group = None
    return f"Page saved for group '{tag}'."
