"""PydanticAI tools for the agentic capture agent.

Provides ``build_capture_tools()`` returning the toolset the agentic
capture agent uses to process multiple text inputs:

- ``get_input(index)`` — return a text input by index.
- ``save_item(index, items)`` — record extracted items for an input.

All tools access their data through ``RunContext[CaptureAgentDeps]``.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext, Tool

from knox.agents.types import CaptureAgentDeps


def build_capture_tools() -> list[Any]:
    """Return the full toolset for the agentic capture agent."""
    return [
        Tool(get_input, takes_ctx=True),
        Tool(save_item, takes_ctx=True),
    ]


async def get_input(ctx: RunContext[CaptureAgentDeps], index: int) -> str:
    """Return a text input by its index for processing."""
    inputs = ctx.deps.inputs
    if index < 0 or index >= len(inputs):
        return f"Input {index} out of range (total: {len(inputs)})."
    ctx.deps.current_input = inputs[index]
    return f"Input {index} of {len(inputs)}:\n{inputs[index]}"


async def save_item(ctx: RunContext[CaptureAgentDeps], index: int, items: list[dict[str, Any]]) -> str:
    """Record extracted knowledge items for an input."""
    ctx.deps.items[index] = items
    ctx.deps.current_input = None
    return f"Saved {len(items)} items for input {index}."
