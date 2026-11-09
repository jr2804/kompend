"""ExtractionAgent — PydanticAI wrapper for the two-phase extraction stage.

Replaces the raw direct LLM call in ``kompend/compilation/two_phase/``
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

from typing import ClassVar, Final

from kompend.agents.base import AgentSpec, KompendAgentBase
from kompend.agents.types import PipelineAgentDeps
from kompend.compilation.schema import ExtractionOutput

SPEC: Final[AgentSpec] = AgentSpec(
    name="extraction",
    result_type=ExtractionOutput,
    deps_type=PipelineAgentDeps,
    template_name="extraction_prompt.md.j2",
    system_template_name="extraction_system.md.j2",
    supports_delta_retry=True,
)


class ExtractionAgent(KompendAgentBase[ExtractionOutput, PipelineAgentDeps]):
    """PydanticAI agent for the compilation extraction stage.

    The definition lives in :data:`SPEC`; the class is the typed facade the
    base class materialises the spec onto.
    """

    spec: ClassVar[AgentSpec | None] = SPEC

    #: Extraction threads one ``message_history`` across every uncached batch
    #: of a source, so a fold at a natural break beats a fold at a threshold.
    compact_on_request: ClassVar[bool] = True
