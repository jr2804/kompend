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
    """

    template_name: str = ""  # set per-rule via _build_agent override
    system_template_name: str = ""  # not used; system prompt is empty
    result_type: type = dict  # type: ignore[arg-type]
    deps_type: type[PipelineAgentDeps] = PipelineAgentDeps
