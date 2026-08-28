"""Leaf-node dependency types for KNOX PydanticAI agents.

These types are kept in a separate module to break circular imports
between the agents package and the compilation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


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
class BaseGenerationDeps:
    """Shared base for all generation-stage runtime contexts.

    Attributes:
        source_id: Provenance / log label for the batch.
        pinned_facts: Facts that must appear in the output (WiCER
            refinement input). Empty list when unused.
        known_titles: Canonical page titles the LLM should use verbatim in
            ``[[wikilinks]]``. Empty list when unknown.
        timeout: Per-call timeout in seconds.
    """

    source_id: str
    timeout: float = 120.0
    pinned_facts: list[str] = field(default_factory=list)
    known_titles: list[str] = field(default_factory=list)


@dataclass
class GenerationDeps(BaseGenerationDeps):
    """Runtime context for the static generation stage.

    Expects concepts_json and content as static prompt variables.
    """

    concepts_json: str = ""
    content: str = ""
    categories: str = ""  # rendered extra-category instructions


@dataclass
class ToolGenerationDeps(BaseGenerationDeps):
    """Runtime context for the agentic generation stage.

    Supports tool-based generation: the agent calls ``fetch_concepts()``
    and ``read_segment()`` tools to discover its work queue.

    Attributes:
        source_content: Full source segment content, loaded via
            the ``read_segment()`` tool on demand.
    """

    source_content: str = ""
    categories: str = ""  # rendered extra-category instructions
    concept_broker: ConceptBrokerProtocol | None = None


@dataclass
class RemediationPageResult:
    """A generated wiki page returned by the agentic remediation agent.

    Submitted via ``deps.pages.append()`` by the ``save_page`` tool.
    """

    name: str
    content: str


@dataclass
class RemediationDeps:
    """Runtime context for the agentic remediation agent.

    The agent discovers its work queue via the ``fetch_gaps()`` tool and
    submits finished pages via ``save_page()`` (collected on ``pages``)
    or declines a gap via ``mark_unfixable(name, reason)``.

    Attributes:
        gaps: Broker over coverage-gap concepts matched to cached
            extraction buckets.
        timeout: Per-call timeout in seconds.
        dry_run: When ``True``, the caller records would-be actions
            instead of writing nodes.
        pages: Pages submitted via ``save_page()`` — populated by the
            tools during the run; the caller applies them afterwards.
    """

    gaps: GapBrokerProtocol
    timeout: float = 120.0
    dry_run: bool = False
    pages: list[RemediationPageResult] = field(default_factory=list)


class ConceptBrokerProtocol(Protocol):
    """Minimal broker surface used by the agentic generation tools.

    Declared locally (instead of importing the concrete
    :class:`knox.compilation.concept_broker.ConceptBroker`) so this leaf
    module keeps zero ``knox.compilation`` imports — importing any
    submodule of that package triggers ``compilation/__init__``, which
    pulls in ``agents.base`` and re-enters the import cycle.
    """

    def fetch(self, caller_id: str, limit: int = 5) -> list[dict[str, Any]]: ...


class GapBrokerProtocol(Protocol):
    """Minimal broker surface used by the agentic remediation tools.

    Mirrors :class:`ConceptBrokerProtocol`'s local-Protocol approach so
    ``agents.types`` stays a zero-``knox.lint``-import leaf module.
    """

    def fetch(self, caller_id: str, limit: int = 5) -> list[dict[str, Any]]: ...

    def mark_done(self, name: str) -> None: ...

    def mark_unfixable(self, name: str, reason: str) -> None: ...


@dataclass
class OverviewDeps:
    """Runtime context for the agentic overview-synthesis agent.

    The agent lists pages via ``list_pages()``, reads full content via
    ``read_page()``, and submits the final overview via ``save_overview()``.

    Attributes:
        pages: Ordered list of page summaries (slug, title, content).
        timeout: Per-call timeout in seconds.
        overview: The submitted overview text, populated by the
            ``save_overview`` tool.
    """

    pages: list[OverviewPage]
    timeout: float = 120.0
    overview: str = ""


@dataclass
class OverviewPage:
    """A single wiki page available to the overview agent."""

    slug: str
    title: str
    content: str


@dataclass
class VerificationVerdict:
    """A single verdict recorded by the agentic verification agent."""

    claim_id: int
    verdict: str  # "supported" | "contradicted" | "unverifiable"
    evidence: str
    sources: list[str] = field(default_factory=list)


@dataclass
class VerificationDeps:
    """Runtime context for the agentic verification agent.

    The agent fetches claims via ``get_claim()``, searches sources via
    ``search_sources()`` and ``read_page()``, and records verdicts via
    ``mark_verified()`` / ``mark_contradicted()`` / ``mark_unverifiable()``.

    Attributes:
        claims: List of claim dicts (id + text).
        sources: List of source snippet dicts (slug + text).
        pages: List of wiki page dicts (slug + content).
        results: Collected verdicts, populated by the mark_* tools.
    """

    claims: list[dict[str, Any]]  # each: {"id": int, "text": str}
    sources: list[dict[str, Any]] = field(default_factory=list)  # each: {"slug", "text"}
    pages: list[dict[str, Any]] = field(default_factory=list)  # each: {"slug", "content"}
    results: list[VerificationVerdict] = field(default_factory=list)
    source_id: str = "verification"
    timeout: float = 120.0


@dataclass
class WriteDeps:
    """Runtime context for the agentic write agent.

    The agent pulls source texts on demand via ``read_source()`` /
    ``list_sources()``, retrieves related titles via ``read_related()``,
    and submits the final page via ``save_page()``.

    Attributes:
        page_title: The page title to write.
        source_texts: List of (stem, text) tuples.
        related_titles: List of related wiki page titles.
        page_content: The agent's submitted page content.
    """

    page_title: str
    source_texts: list[tuple[str, str]]  # (stem, text)
    related_titles: list[str] = field(default_factory=list)
    page_content: str = ""
    source_id: str = "write"
    timeout: float = 120.0


@dataclass
class ContextChunkDeps:
    """Runtime context for the agentic context-chunk enrichment agent.

    The agent iterates over chunks and enriches each with situating context.
    Final context strings are collected in ``results`` keyed by chunk_id.

    Attributes:
        doc_title: The document title.
        doc_summary: Brief summary of the document (for context injection).
        chunks: List of chunk dicts (id, heading_chain, text).
        results: Collected context strings keyed by chunk_id.
        source_id: Identifier for the document being chunked.
        timeout: Per-call timeout in seconds.
    """

    doc_title: str
    doc_summary: str
    chunks: list[dict[str, Any]]  # each: {"id": int, "heading_chain": list[str], "text": str}
    results: dict[int, str] = field(default_factory=dict)
    source_id: str = "context_chunk"
    timeout: float = 120.0
    current_chunk: dict[str, Any] | None = None

@dataclass
class SummarizerDeps:
    """Runtime context for the agentic summarizer agent.

    The agent iterates over internal chunk nodes and writes a summary for each.

    Attributes:
        nodes: List of node dicts (id, heading_chain, content).
        results: Collected summaries keyed by node id.
        source_id: Identifier for the tree being summarized.
        timeout: Per-call timeout in seconds.
    """

    nodes: list[dict[str, Any]]  # each: {"id": str, "heading_chain": str, "content": str}
    results: dict[str, str] = field(default_factory=dict)
    source_id: str = "summarizer"
    timeout: float = 120.0
    current_node: dict[str, Any] | None = None

@dataclass
class ExtractionAgentDeps:
    """Runtime context for the agentic extraction agent.

    The agent iterates over source sections and extracts data from each.

    Attributes:
        source: Identifier for the source being extracted.
        sections: List of section text strings.
        extractions: Collected extraction JSON keyed by section index.
        source_id: Identifier for logging/tracking.
        timeout: Per-call timeout in seconds.
    """

    source: str
    sections: list[str]
    extractions: dict[int, str] = field(default_factory=dict)
    source_id: str = "extraction"
    timeout: float = 120.0
    current_section: str | None = None

@dataclass
class CategorizeAgentDeps:
    """Runtime context for the agentic categorize agent.

    The agent iterates over source file batches and builds a category tree.

    Attributes:
        files: List of file dicts (filename, summary, tags).
        categories: Collected category-tree JSON built incrementally.
        source_id: Identifier for the organize run.
        timeout: Per-call timeout in seconds.
    """

    files: list[dict[str, Any]]  # each: {"filename": str, "summary": str, "tags": list}
    categories: list[dict[str, Any]] = field(default_factory=list)
    unfiled: list[str] = field(default_factory=list)
    source_id: str = "categorize"
    timeout: float = 120.0
    max_depth: int = 3
    min_subcategories: int = 2
    min_files_per_category: int = 2

