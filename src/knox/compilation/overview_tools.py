"""PydanticAI tools for the agentic overview-synthesis agent.

Provides ``build_overview_tools()`` returning the toolset an agentic
overview agent uses to write ``overview.md``:

- ``list_pages(filter)`` — list page summaries (slug + title), optionally
  filtered by substring.
- ``read_page(slug)`` — read a page's full content.
- ``save_overview(content)`` — submit the final overview Markdown.

All tools access their data through ``RunContext[OverviewDeps]``. The
submitted overview is collected on ``deps.overview`` and written by the
caller after the agent run finishes.
"""

from __future__ import annotations

from pydantic_ai import RunContext, Tool

from knox.agents.types import OverviewDeps


def build_overview_tools() -> list[Tool[OverviewDeps]]:
    """Return the PydanticAI toolset for the agentic overview agent."""
    return [
        Tool(list_pages),
        Tool(read_page),
        Tool(save_overview),
    ]


async def list_pages(ctx: RunContext[OverviewDeps], filter: str | None = None) -> str:
    """List the wiki pages available for the overview.

    Args:
        ctx: Runtime context with the page list via deps.
        filter: Optional substring to filter page titles by.

    Returns:
        A newline-separated list of ``slug: title`` lines.
    """
    pages = ctx.deps.pages
    if filter:
        pages = [p for p in pages if filter.lower() in p.title.lower()]
    if not pages:
        return "(no pages match)"
    return "\n".join(f"{p.slug}: {p.title}" for p in pages)


async def read_page(ctx: RunContext[OverviewDeps], slug: str) -> str:
    """Read the full content of one wiki page.

    Args:
        ctx: Runtime context with the page list via deps.
        slug: Page slug from list_pages().

    Returns:
        The page's full Markdown content, or an error string when the
        slug is unknown.
    """
    for page in ctx.deps.pages:
        if page.slug == slug:
            return page.content
    return f"Unknown page: {slug}"


async def save_overview(ctx: RunContext[OverviewDeps], content: str) -> str:
    """Submit the final overview Markdown.

    Args:
        ctx: Runtime context; the overview is collected on deps.
        content: Full overview Markdown (a ``# Wiki Overview`` heading
            is added by the caller).

    Returns:
        Confirmation string.
    """
    ctx.deps.overview = content
    return "Overview saved."
