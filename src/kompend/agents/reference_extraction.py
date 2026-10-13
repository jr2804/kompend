"""ReferenceExtractionAgent — PydanticAI wrapper for bibliographic reference parsing.

Follows the same pattern as :class:`~kompend.agents.extraction.ExtractionAgent`.
The agent's ``output_type`` is
:class:`~kompend.core.models.references.ReferenceExtractionOutput`, so the LLM
must produce a validated list of :class:`~kompend.core.models.references.Reference`
objects.

Usage::

    from kompend.agents.reference_extraction import ReferenceExtractionAgent

    agent = ReferenceExtractionAgent(model_string="ollama/gpt-oss:20b")
    result = agent.run(PipelineAgentDeps(text=ref_section, source_id=stem))
    for ref in result.references:
        print(ref.raw, ref.doi)
"""

from __future__ import annotations

from typing import ClassVar

from kompend.agents.base import KompendAgentBase
from kompend.agents.deps import PipelineAgentDeps
from kompend.core.models.references import ReferenceExtractionOutput


class ReferenceExtractionAgent(KompendAgentBase[ReferenceExtractionOutput, PipelineAgentDeps]):
    """PydanticAI agent for extracting bibliographic references.

    The agent's system + user prompts come from Jinja2 templates
    (``reference_extraction_system.md.j2``,
    ``reference_extraction_prompt.md.j2``) under ``compilation/``.
    """

    template_name: ClassVar[str] = "reference_extraction_prompt.md.j2"
    system_template_name: ClassVar[str] = "reference_extraction_system.md.j2"
