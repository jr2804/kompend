"""Dependency types for KNOX PydanticAI agents.

PydanticAI passes a typed ``Deps`` object to every tool call and into the
system prompt template. The :class:`PipelineAgentDeps` carries the
runtime context an agent needs to do its job — the source text being
analysed, the source identifier for provenance, and per-call budget
hints from the surrounding pipeline.

Why a single ``Deps`` type instead of one per agent? It keeps the
agent-factory surface small. Agents that need additional context
(generation, wicer) extend the base class with their own Deps subclass
in a follow-up phase. Phase 1 (knox-pyda) only needs extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineAgentDeps:
    """Runtime context shared by all pipeline agents.

    Attributes:
        text: The source content the agent is operating on. For
            extraction this is the raw chunk text; for generation it
            would be the concept batch; etc. Each agent decides how
            to use it.
        source_id: Stable identifier for the source (file path stem,
            URL, etc.) used for provenance and logging. Does not need
            to be a Path — plain string is fine.
        timeout: Per-call timeout in seconds. Forwarded to
            PydanticAI's model settings so a single slow call doesn't
            stall the pipeline.
    """

    text: str
    source_id: str
    timeout: float = 120.0


@dataclass
class GenerationDeps:
    """Runtime context for the compilation generation stage.

    Unlike :class:`PipelineAgentDeps` (extraction), the generation
    stage produces wiki pages from a batch of concepts plus source
    segments. The prompt template ``generation_prompt.md.j2`` expects
    three variables: ``concepts_json``, ``content``, and optional
    ``pinned_facts``.

    Attributes:
        concepts_json: The concept batch encoded via
            :func:`knox.llm.format.encode_for_llm` in the configured
            data format.
        content: Combined source-content block (one ``### label`` section
            per source segment).
        source_id: Provenance / log label for the batch.
        pinned_facts: Facts that must appear in the output (WiCER
            refinement input). Empty list when unused.
        timeout: Per-call timeout in seconds.
    """

    concepts_json: str
    content: str
    source_id: str
    pinned_facts: list[str] = field(default_factory=list)
    timeout: float = 120.0
