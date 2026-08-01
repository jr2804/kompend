"""WriteAgent — PydanticAI agent for wiki page synthesis (knox write)."""

from __future__ import annotations

from typing import ClassVar

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps


class WriteAgent(KnoxAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent that synthesises source texts into a wiki page.

    Returns free-text Markdown. Used by the ``knox write`` command to
    generate a continuous-prose wiki page from extracted source texts.
    The prompt is rendered by the caller and passed via ``deps.text``,
    so ``render_user_prompt`` is overridden to return it directly.
    """

    template_name: ClassVar[str] = ""  # not used; render_user_prompt is overridden
    system_template_name: ClassVar[str] = ""  # no system prompt needed

    def render_user_prompt(self, deps: PipelineAgentDeps) -> str:  # noqa: PLR6301
        """Return the pre-rendered prompt from deps.text."""
        return deps.text
