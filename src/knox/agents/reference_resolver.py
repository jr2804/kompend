"""ReferenceResolverAgent — PydanticAI wrapper for DOI/URL resolution.

Follows the same pattern as :class:`~knox.agents.reference_extraction.ReferenceExtractionAgent`.
The agent's ``output_type`` is
:class:`~knox.core.models.references.ReferenceResolutionOutput`, so the LLM
must return a single ``{resolved, doi, url, note}`` object per reference.

This powers the ``knox references fetch`` subcommand: it takes a reference's
raw citation text and attempts to recover a canonical DOI or URL so the
stored reference becomes resolvable from the wiki's reference links.

Usage::

    from knox.agents.reference_resolver import ReferenceResolverAgent

    agent = ReferenceResolverAgent(model_string="ollama/gpt-oss:20b")
    result = agent.run(PipelineAgentDeps(text=ref.raw, source_id=ref.citing_source))
    if result.resolved and result.doi:
        ...
"""

from __future__ import annotations

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps
from knox.core.models.references import ReferenceResolutionOutput


class ReferenceResolverAgent(KnoxAgentBase[ReferenceResolutionOutput, PipelineAgentDeps]):
    """PydanticAI agent for resolving missing DOIs/URLs of references.

    The agent's system + user prompts come from Jinja2 templates
    (``reference_resolver_system.md.j2``,
    ``reference_resolver_prompt.md.j2``) under ``compilation/``.
    """

    template_name: str = "reference_resolver_prompt.md.j2"
    system_template_name: str = "reference_resolver_system.md.j2"
    result_type: type[ReferenceResolutionOutput] = ReferenceResolutionOutput
    deps_type: type[PipelineAgentDeps] = PipelineAgentDeps
    temperature: float | None = 0.0
