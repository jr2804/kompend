"""PydanticAI tools for the agentic categorizer agent.

Provides ``build_categorize_tools()`` returning the toolset the agentic
categorizer uses to build a category tree incrementally:

- ``get_file_batch(start, count)`` — return a batch of files to categorize.
- ``add_category(path, files)`` — record a category with its files.

All tools access their data through ``RunContext[CategorizeAgentDeps]``.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic_ai import RunContext

from knox.agents.types import CategorizeAgentDeps


async def get_file_batch(ctx: RunContext[CategorizeAgentDeps], start: int, count: int = 10) -> str:
    """Return a batch of files for categorization as JSON."""
    files = ctx.deps.files[start:start+count]
    if not files:
        return f"No more files. Total categorized: {len(ctx.deps.categories)}."
    batch_json = json.dumps({"files": files, "position": start, "total": len(ctx.deps.files)})
    return f"Files {start}-{start+len(files)} of {len(ctx.deps.files)}:\n{batch_json}"


async def add_category(ctx: RunContext[CategorizeAgentDeps], path: str, files: list[str]) -> str:
    """Record a category and its files in the tree."""
    category = {"path": path, "files": files}
    ctx.deps.categories.append(category)
    for f in files:
        if f not in ctx.deps.unfiled or f in ctx.deps.unfiled:
            ctx.deps.unfiled = [u for u in ctx.deps.unfiled if u != f]
    return f"Category '{path}' added with {len(files)} files."


def build_categorize_tools() -> list[Any]:
    """Return the full toolset for the agentic categorize agent."""
    return [get_file_batch, add_category]
