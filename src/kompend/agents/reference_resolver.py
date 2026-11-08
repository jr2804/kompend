"""ReferenceResolverAgent — PydanticAI wrapper for DOI/URL resolution.

Follows the same pattern as :class:`~kompend.agents.reference_extraction.ReferenceExtractionAgent`.
The agent's ``output_type`` is
:class:`~kompend.core.models.references.ReferenceResolutionOutput`, so the LLM
must return a single ``{resolved, doi, url, note}`` object per reference.

This powers the ``kompend references fetch`` subcommand: it takes a reference's
raw citation text and attempts to recover a canonical DOI or URL so the
stored reference becomes resolvable from the wiki's reference links.

Usage::

    from kompend.agents.reference_resolver import ReferenceResolverAgent

    agent = ReferenceResolverAgent(model_string="ollama/gpt-oss:20b")
    result = agent.run(PipelineAgentDeps(text=ref.raw, source_id=ref.citing_source))
    if result.resolved and result.doi:
        ...
"""

from __future__ import annotations

from typing import ClassVar, Final

from kompend.agents.base import AgentSpec, KompendAgentBase
from kompend.agents.deps import PipelineAgentDeps
from kompend.core.models.references import ReferenceResolutionOutput

SPEC: Final[AgentSpec] = AgentSpec(
    name="reference_resolver",
    result_type=ReferenceResolutionOutput,
    deps_type=PipelineAgentDeps,
    template_name="reference_resolver_prompt.md.j2",
    system_template_name="reference_resolver_system.md.j2",
    temperature=0.0,
)


class ReferenceResolverAgent(KompendAgentBase[ReferenceResolutionOutput, PipelineAgentDeps]):
    """PydanticAI agent for resolving missing DOIs/URLs of references.

    The agent's system + user prompts come from Jinja2 templates
    (``reference_resolver_system.md.j2``,
    ``reference_resolver_prompt.md.j2``) under ``compilation/``.
    """

    spec: ClassVar[AgentSpec | None] = SPEC
    temperature: float | None = 0.0
