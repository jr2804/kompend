"""WriteAgent — PydanticAI agent for wiki page synthesis (knox write)."""

from __future__ import annotations

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps


class WriteAgent(KnoxAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent that synthesises source texts into a wiki page.

    Returns free-text Markdown. Used by the ``knox write`` command to
    generate a continuous-prose wiki page from extracted source texts.
    The prompt is rendered by the caller and passed via ``deps.text``,
    so ``render_user_prompt`` is overridden to return it directly.
    """

    template_name: str = ""  # not used; render_user_prompt is overridden
    system_template_name: str = ""  # no system prompt needed
    result_type: type = str
    deps_type: type[PipelineAgentDeps] = PipelineAgentDeps

    def render_user_prompt(self, deps: PipelineAgentDeps) -> str:
        """Return the pre-rendered prompt from deps.text."""
        return deps.text
