"""PydanticAI tools for the agentic context-chunk enrichment agent.

Provides ``build_context_chunk_tools()`` returning the toolset the
agentic context chunk agent uses to enrich chunks with situating context:

- ``next_chunk()`` — return the next chunk that needs context.
- ``situate_chunk(chunk_id, content)`` — record the context for a chunk.

All tools access their data through ``RunContext[ContextChunkDeps]``.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext

from kompend.agents.types import ContextChunkDeps

#: Chunk text shown to the context-chunk agent — in the tool payload and in the
#: prompt it answers. One budget for both, so the agent never sees a longer
#: chunk in the tool result than the prompt prepared it for.
CONTEXT_CHUNK_TEXT_PREVIEW_CHARS: int = 500


def build_context_chunk_tools() -> list[Any]:
    """Return the full toolset for the agentic context chunk agent."""
    return [next_chunk, situate_chunk]


async def next_chunk(ctx: RunContext[ContextChunkDeps]) -> str:
    """Return the next chunk that still needs situating context."""
    processed = set(ctx.deps.results.keys())
    pending = [c for c in ctx.deps.chunks if c["id"] not in processed]
    if not pending:
        return "No more chunks to process."
    chunk = pending[0]
    ctx.deps.current_chunk = chunk
    stem = chunk["heading_chain"]
    heading = " > ".join(stem) if stem else "document root"
    text = chunk["text"][:CONTEXT_CHUNK_TEXT_PREVIEW_CHARS]
    return f"Chunk {chunk['id']}: heading={heading}, text={text}"


async def situate_chunk(ctx: RunContext[ContextChunkDeps], chunk_id: int, content: str) -> str:
    """Record the situating context for a chunk."""
    ctx.deps.results[chunk_id] = content.strip()
    ctx.deps.current_chunk = None
    return f"Context recorded for chunk {chunk_id}."
