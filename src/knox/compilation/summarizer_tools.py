"""PydanticAI tools for the agentic summarizer agent.

Provides ``build_summarizer_tools()`` returning the toolset the agentic
summarizer uses to walk internal chunk nodes:

- ``next_node()`` — return the next node that needs a summary.
- ``summarize_node(node_id, summary)`` — record the summary for a node.

All tools access their data through ``RunContext[SummarizerDeps]``.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext

from knox.agents.types import SummarizerDeps


def build_summarizer_tools() -> list[Any]:
    """Return the full toolset for the agentic summarizer agent."""
    return [next_node, summarize_node]


async def next_node(ctx: RunContext[SummarizerDeps]) -> str:
    """Return the next node that still needs a summary."""
    processed = set(ctx.deps.results.keys())
    pending = [n for n in ctx.deps.nodes if n["id"] not in processed]
    if not pending:
        return "No more nodes to summarize."
    node = pending[0]
    ctx.deps.current_node = node
    heading = node.get("heading_chain", "") or "document root"
    content = node.get("content", "")[:2000]
    return f"Node {node['id']}: heading={heading}, content={content}"


async def summarize_node(ctx: RunContext[SummarizerDeps], node_id: str, summary: str) -> str:
    """Record the summary for a node."""
    ctx.deps.results[node_id] = summary.strip()
    ctx.deps.current_node = None
    return f"Summary recorded for node {node_id}."
