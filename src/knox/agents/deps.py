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
        extra_context: Optional structured context string injected
            as a system message before the user prompt (used by the
            AGENTIC extraction strategy to inject the memory digest).
            Empty by default; ignored by non-agentic strategies.
    """

    text: str
    source_id: str
    timeout: float = 120.0
    extra_context: str = ""


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
        known_titles: Canonical page titles the LLM should use verbatim in
            ``[[wikilinks]]`` (all merged concept names plus canonicals
            learned from the previous compile's alias map). Empty list when
            unknown.
        timeout: Per-call timeout in seconds.
    """

    concepts_json: str
    content: str
    source_id: str
    pinned_facts: list[str] = field(default_factory=list)
    categories: str = ""  # rendered extra-category instructions ("- \"name\" — desc" lines)
    known_titles: list[str] = field(default_factory=list)
    timeout: float = 120.0


@dataclass
class QueryDeps:
    """Runtime context for the query/synthesis stage.

    The query agent answers a user question based on retrieved context
    chunks. The system prompt template ``query/synthesize_system.md.j2``
    expects ``context`` and ``verbosity`` variables; the user prompt is
    the raw question string.

    Attributes:
        question: The user's free-text question.
        context: Pre-formatted context block (one entry per search
            result, with title, source, and snippet).
        source_id: Log/provenance label for the query.
        verbosity: Synthesis verbosity (``brief``, ``standard``,
            ``comprehensive``). Controls answer length.
        timeout: Per-call timeout in seconds.
    """

    question: str
    context: str
    source_id: str
    verbosity: str = "standard"
    timeout: float = 120.0


@dataclass
class WicerProbeDeps:
    """Runtime context for the WiCER probe-generation stage.

    The probe agent generates Q&A probes from source text to test the
    compiled wiki against. The prompt template ``wicer_probe.md.j2``
    expects one variable: ``text``.

    Attributes:
        text: The combined source text to generate probes from.
        source_id: Log/provenance label for the batch.
        timeout: Per-call timeout in seconds.
    """

    text: str
    source_id: str
    timeout: float = 120.0


@dataclass
class WicerEvaluateDeps:
    """Runtime context for the WiCER evaluate stage.

    The evaluate agent checks if the compiled wiki correctly answers
    the generated probes. The prompt template ``wicer_evaluate.md.j2``
    expects two variables: ``wiki_text`` and ``probes_json``.

    Attributes:
        wiki_text: A segment of compiled wiki text to evaluate against.
        probes_json: The probes encoded via :func:`knox.llm.format.encode_for_llm`.
        source_id: Log/provenance label.
        timeout: Per-call timeout in seconds.
    """

    wiki_text: str
    probes_json: str
    source_id: str
    timeout: float = 120.0


@dataclass
class WicerDiagnoseDeps:
    """Runtime context for the WiCER diagnose stage.

    The diagnose agent extracts critical facts from evaluation failures
    so they can be re-injected as pinned facts in the next compilation
    round. The prompt template ``wicer_diagnose.md.j2`` expects one
    variable: ``failures_json``.

    Attributes:
        failures_json: The failed evaluations encoded via
            :func:`knox.llm.format.encode_for_llm`.
        source_id: Log/provenance label.
        timeout: Per-call timeout in seconds.
    """

    failures_json: str
    source_id: str
    timeout: float = 120.0


@dataclass
class CategorizeDeps:
    """Runtime context for the ``knox organize`` categorisation stage.

    The categorise agent proposes a hierarchical directory tree for the
    files in ``sources/``. The prompt template
    ``organize/categorize_prompt.md.j2`` expects ``files_json``, and the
    system template ``organize/categorize_system.md.j2`` expects the
    three tree-constraint values (``max_depth``, ``min_subcategories``,
    ``min_files_per_category``).

    Attributes:
        files_json: Serialised file/summary/tags payload for the prompt
            (built by :func:`knox.organize.categorize._build_files_payload`).
        max_depth: Maximum category tree depth.
        min_subcategories: Minimum sibling subcategories for a non-leaf
            category.
        min_files_per_category: Minimum files a leaf category must hold.
        source_id: Log/provenance label (always ``"categorize"``).
        timeout: Per-call timeout in seconds.
    """

    files_json: str
    max_depth: int = 3
    min_subcategories: int = 2
    min_files_per_category: int = 2
    source_id: str = "categorize"
    timeout: float = 120.0
