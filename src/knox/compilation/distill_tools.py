"""PydanticAI tools for the agentic distill agent.

Provides ``build_distill_tools()`` returning the toolset the agentic
distill agent uses to process tag groups:

- ``get_group(tag)`` — return the notes for a tag group.
- ``save_page(tag, page)`` — record a distilled wiki page for a group.

All tools access their data through ``RunContext[DistillAgentDeps]``.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic_ai import RunContext, Tool

from knox.agents.types import DistillAgentDeps


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
            notes_json = json.dumps(g["notes"])
            return f"Group '{tag}' with {len(g['notes'])} notes:\n{notes_json}"
    return f"Group '{tag}' not found. Available: {[g['tag'] for g in ctx.deps.groups]}"


async def save_page(ctx: RunContext[DistillAgentDeps], tag: str, page: dict[str, Any]) -> str:
    """Record a distilled wiki page for a tag group."""
    ctx.deps.pages[tag] = page
    ctx.deps.current_group = None
    return f"Page saved for group '{tag}'."
