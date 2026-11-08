"""ExtractionReduceAgent — merges partial per-chunk extractions into one.

Used by the ``reduce`` extraction strategy. Replaces the naive
``_merge_extractions`` string-concat with an LLM call that synthesises
a coherent document-level summary and deduplicates entities/concepts/tags.
"""

from __future__ import annotations

from typing import ClassVar, Final

from kompend.agents.base import AgentSpec, KompendAgentBase
from kompend.agents.types import PipelineAgentDeps
from kompend.compilation.schema import ExtractionOutput

SPEC: Final[AgentSpec] = AgentSpec(
    name="extraction_reduce",
    result_type=ExtractionOutput,
    deps_type=PipelineAgentDeps,
    template_name="extraction_reduce_prompt.md.j2",
    system_template_name="extraction_reduce_system.md.j2",
)


class ExtractionReduceAgent(KompendAgentBase[ExtractionOutput, PipelineAgentDeps]):
    """PydanticAI agent that merges partial extractions into a document-level result.

    Receives a JSON-serialised array of partial ``ExtractionOutput`` objects
    and returns a single merged ``ExtractionOutput``. The definition lives in
    :data:`SPEC`.
    """

    spec: ClassVar[AgentSpec | None] = SPEC
