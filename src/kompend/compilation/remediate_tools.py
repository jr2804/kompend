"""PydanticAI tools for the agentic remediation agent.

Provides ``build_remediation_tools()`` returning the toolset an agentic
remediation agent uses to work a coverage-gap queue:

- ``fetch_gaps(limit)`` — next batch of gap concepts (matched to cached
  extraction buckets: name, definitions, sources, summaries).
- ``save_page(name, content)`` — submit a generated wiki page.
- ``mark_unfixable(name, reason)`` — decline a gap with a reason.

All tools access their data through ``RunContext[RemediationDeps]``.
Page submissions are collected in ``deps.pages`` by the tools and applied
by the caller after the agent run finishes — the tools never touch the
wiki directly.
"""

from __future__ import annotations

from pydantic_ai import RunContext, Tool

from kompend.agents.types import RemediationDeps, RemediationPageResult
from kompend.core.config._configs import LlmDataFormat
from kompend.llm.format import encode_for_llm


def build_remediation_tools() -> list[Tool[RemediationDeps]]:
    """Return the PydanticAI toolset for the agentic remediation agent."""
    return [
        Tool(fetch_gaps),
        Tool(save_page),
        Tool(mark_unfixable),
    ]


async def fetch_gaps(ctx: RunContext[RemediationDeps], limit: int = 5) -> str:
    """Fetch the next batch of coverage-gap concepts to generate pages for.

    Args:
        ctx: Runtime context with the gap broker via deps.
        limit: Maximum number of concepts to return (default 5, max 20).

    Returns:
        A JSON-encoded list of concept buckets, each with keys
        ``name``, ``definitions``, ``sources``, ``summaries``.
        Returns ``[]`` when the queue is exhausted.
    """
    batch = ctx.deps.gaps.fetch("remediate", limit=max(min(limit, 20), 1))
    if not batch:
        return "[]"
    return encode_for_llm(batch, LlmDataFormat.JSON)


async def save_page(ctx: RunContext[RemediationDeps], name: str, content: str) -> str:
    """Submit a generated wiki page for a coverage-gap concept.

    Args:
        ctx: Runtime context; completed pages are collected on deps.
        name: Exact concept name the page covers.
        content: Full wiki page Markdown.

    Returns:
        Confirmation string.
    """
    ctx.deps.gaps.mark_done(name)
    ctx.deps.pages.append(RemediationPageResult(name=name, content=content))
    return f"Saved page for '{name}'."


async def mark_unfixable(ctx: RunContext[RemediationDeps], name: str, reason: str) -> str:
    """Decline to generate a page for a gap concept, with a reason.

    Args:
        ctx: Runtime context with the gap broker via deps.
        name: Exact concept name being declined.
        reason: Short explanation (e.g. "insufficient source material").

    Returns:
        Confirmation string.
    """
    ctx.deps.gaps.mark_unfixable(name, reason)
    return f"Marked '{name}' unfixable: {reason}"
