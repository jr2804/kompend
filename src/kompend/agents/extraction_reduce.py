"""ExtractionReduceAgent — merges partial per-chunk extractions into one.

Used by the ``reduce`` extraction strategy. Replaces the naive
``_merge_extractions`` string-concat with an LLM call that synthesises
a coherent document-level summary and deduplicates entities/concepts/tags.
"""

from __future__ import annotations

from typing import ClassVar

from kompend.agents.base import KompendAgentBase
from kompend.agents.types import PipelineAgentDeps
from kompend.compilation.schema import ExtractionOutput


class ExtractionReduceAgent(KompendAgentBase[ExtractionOutput, PipelineAgentDeps]):
    """PydanticAI agent that merges partial extractions into a document-level result.

    Receives a JSON-serialised array of partial ``ExtractionOutput`` objects
    and returns a single merged ``ExtractionOutput``.
    """

    template_name: ClassVar[str] = "extraction_reduce_prompt.md.j2"
    system_template_name: ClassVar[str] = "extraction_reduce_system.md.j2"
    output_budget_key: ClassVar[str] = "extract_output_max"
