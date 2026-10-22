"""ExtractionAgent — PydanticAI wrapper for the two-phase extraction stage.

Replaces the raw direct LLM call in ``kompend/compilation/_two_phase_core.py``
with a typed PydanticAI agent. The agent's ``output_type`` is
:class:`~kompend.compilation.schema.ExtractionOutput`, so the LLM is
asked to produce that exact Pydantic model. PydanticAI handles JSON
extraction + validation + auto-retry on mismatch internally.

Usage::

    from kompend.agents.extraction import ExtractionAgent

    agent = ExtractionAgent(model_string="ollama/gpt-oss:20b")
    result = agent.run(PipelineAgentDeps(text=chunk, source_id=stem))
    print(result.summary, result.entities, result.concepts)
"""

from __future__ import annotations

from typing import ClassVar

from kompend.agents.base import KompendAgentBase
from kompend.agents.types import PipelineAgentDeps
from kompend.compilation.schema import ExtractionOutput


class ExtractionAgent(KompendAgentBase[ExtractionOutput, PipelineAgentDeps]):
    """PydanticAI agent for the compilation extraction stage.

    The agent's system + user prompts come from the existing Jinja2
    templates (``extraction_system.md.j2``, ``extraction_prompt.md.j2``)
    so the LLM-facing surface is identical to the current direct LLM
    path. Only the dispatch and validation are upgraded.
    """

    template_name: ClassVar[str] = "extraction_prompt.md.j2"
    system_template_name: ClassVar[str] = "extraction_system.md.j2"
    supports_delta_retry: ClassVar[bool] = True
    output_budget_key: ClassVar[str] = "extract_output_max"
