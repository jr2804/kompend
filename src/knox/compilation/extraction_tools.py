"""PydanticAI tools for the agentic extraction agent.

Provides ``build_extraction_tools()`` returning the toolset the agentic
extractor uses to walk source sections:

- ``get_section(section_id)`` — return the text of a section by id.
- ``save_extraction(section_id, data)`` — record extracted data for a section.

All tools access their data through ``RunContext[ExtractionAgentDeps]``.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext

from knox.agents.types import ExtractionAgentDeps


async def get_section(ctx: RunContext[ExtractionAgentDeps], section_id: int) -> str:
    """Return the text of a source section by its index."""
    sections = ctx.deps.sections
    if section_id < 0 or section_id >= len(sections):
        return f"Section {section_id} out of range (total: {len(sections)})."
    ctx.deps.current_section = sections[section_id]
    return f"Section {section_id}:\n{sections[section_id]}"


async def save_extraction(ctx: RunContext[ExtractionAgentDeps], section_id: int, data: str) -> str:
    """Record extracted data for a section."""
    ctx.deps.extractions[section_id] = data.strip()
    ctx.deps.current_section = None
    return f"Extraction saved for section {section_id}."


def build_extraction_tools() -> list[Any]:
    """Return the full toolset for the agentic extraction agent."""
    return [get_section, save_extraction]
