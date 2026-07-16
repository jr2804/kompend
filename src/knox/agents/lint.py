"""LintRuleAgent — PydanticAI agent for LLM-powered lint rules."""

from __future__ import annotations

from typing import Any

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps


class LintRuleAgent(KnoxAgentBase[dict[str, Any], PipelineAgentDeps]):
    """PydanticAI agent for LLM-powered lint rules.

    Returns free-form JSON (dict) matching the rule's expected schema.
    Each rule (factual-consistency, clarity-score, coverage-gap,
    contradiction-detection) has its own template and output schema.
    The prompt is rendered by the rule itself and passed via ``deps.text``.
    """

    template_name: str = ""  # not used; render_user_prompt is overridden
    system_template_name: str = ""  # not used; system prompt is empty
    result_type: type = dict  # type: ignore[arg-type]
    deps_type: type[PipelineAgentDeps] = PipelineAgentDeps

    def render_user_prompt(self, deps: PipelineAgentDeps) -> str:
        """Return the pre-rendered prompt from deps.text.

        Lint rules render their own templates in ``_call_agent()`` and
        pass the result as ``deps.text``, so the base class template
        rendering is bypassed.
        """
        return deps.text
