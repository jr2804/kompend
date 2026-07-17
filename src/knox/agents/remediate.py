"""RemediationAgent — PydanticAI agent for lint-driven concept-page generation."""

from __future__ import annotations

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps


class RemediationAgent(KnoxAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent that generates a wiki concept page from extraction data.

    Used by tier-2 remediation (``lint/remediate_concept.md.j2``) to fill
    coverage gaps. Returns free-text Markdown — distinct from
    :class:`LintRuleAgent`, which returns structured JSON for lint checks.

    The prompt is rendered by the remediation logic and passed via
    ``deps.text``, so ``render_user_prompt`` is overridden to return it
    directly.
    """

    template_name: str = ""  # not used; render_user_prompt is overridden
    system_template_name: str = ""  # not used; system prompt is empty
    result_type: type = str
    deps_type: type[PipelineAgentDeps] = PipelineAgentDeps

    def render_user_prompt(self, deps: PipelineAgentDeps) -> str:  # noqa: PLR6301
        """Return the pre-rendered prompt from deps.text."""
        return deps.text
