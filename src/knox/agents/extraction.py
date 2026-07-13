"""ExtractionAgent — PydanticAI wrapper for the two-phase extraction stage.

Replaces the raw litellm call in ``knox/compilation/_two_phase_core.py``
with a typed PydanticAI agent. The agent's ``output_type`` is
:class:`~knox.compilation.schema.ExtractionOutput`, so the LLM is
asked to produce that exact Pydantic model. PydanticAI handles JSON
extraction + validation + auto-retry on mismatch internally.

Usage::

    from knox.agents.extraction import ExtractionAgent

    agent = ExtractionAgent(litellm_model="ollama/gpt-oss:20b")
    result = agent.run(PipelineAgentDeps(text=chunk, source_id=stem))
    print(result.summary, result.entities, result.concepts)
"""

from __future__ import annotations

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps
from knox.compilation.schema import ExtractionOutput


class ExtractionAgent(KnoxAgentBase[ExtractionOutput, PipelineAgentDeps]):
    """PydanticAI agent for the compilation extraction stage.

    The agent's system + user prompts come from the existing Jinja2
    templates (``extraction_system.md.j2``, ``extraction_prompt.md.j2``)
    so the LLM-facing surface is identical to the current litellm
    path. Only the dispatch and validation are upgraded.
    """

    template_name: str = "extraction_prompt.md.j2"
    system_template_name: str = "extraction_system.md.j2"
    result_type: type[ExtractionOutput] = ExtractionOutput
    deps_type: type[PipelineAgentDeps] = PipelineAgentDeps
