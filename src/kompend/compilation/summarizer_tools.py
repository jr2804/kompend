"""PydanticAI tools for the agentic summarizer agent.

Provides ``build_summarizer_tools()`` returning the toolset the agentic
summarizer uses to walk internal chunk nodes:

- ``next_node()`` — return the next node that needs a summary.
- ``summarize_node(node_id, summary)`` — record the summary for a node.

All tools access their data through ``RunContext[SummarizerDeps]``.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import ModelRetry, RunContext

from kompend.agents.types import SummarizerDeps

#: Node content handed to the summarizer agent's tool payload. Content longer
#: than this is split into several payload windows by ``NodeSummarizer``, so
#: the model always sees the whole node — one window per call, never a
#: silent truncation.
NODE_CONTENT_CHARS: int = 2000


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
    # The payload is pre-windowed by ``NodeSummarizer`` (one window per
    # payload entry), so the content is handed over in full — never
    # truncated here.
    content = node.get("content", "")
    return f"node_id: {node['id']}\nheading: {heading}\ncontent: {content}"


async def summarize_node(ctx: RunContext[SummarizerDeps], node_id: str, summary: str) -> str:
    """Record the summary for a node.

    The id must be one of the session payload nodes; a small model that
    mangles it gets a correction instead of writing a summary nobody reads.
    """
    node_id = node_id.strip()
    known = {node["id"] for node in ctx.deps.nodes}
    if node_id not in known:
        raise ModelRetry(f"Unknown node id {node_id!r}. Call next_node() and pass the id it returns verbatim. Nodes awaiting a summary: {sorted(known)}")
    ctx.deps.results[node_id] = summary.strip()
    ctx.deps.current_node = None
    return f"Summary recorded for node {node_id}."
