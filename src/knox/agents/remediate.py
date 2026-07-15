"""RemediationAgent — PydanticAI agent for lint-driven concept-page generation."""

from __future__ import annotations

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps


class RemediationAgent(KnoxAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent that generates a wiki concept page from extraction data.

    Used by tier-2 remediation (``lint/remediate_concept.md.j2``) to fill
    coverage gaps. Returns free-text Markdown — distinct from
    :class:`LintRuleAgent`, which returns structured JSON for lint checks.
    """

    template_name: str = "lint/remediate_concept.md.j2"
    system_template_name: str = ""  # system prompt is empty for this agent
    result_type: type = str
    deps_type: type[PipelineAgentDeps] = PipelineAgentDeps
