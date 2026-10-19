"""Leaf-node dependency types for kompend PydanticAI agents.

These types are kept in a separate module to break circular imports
between the agents package and the compilation pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict


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
        source_id: Provenance / log label for the remediation run.
        timeout: Per-call timeout in seconds.
        dry_run: When ``True``, the caller records would-be actions
            instead of writing nodes.
        pages: Pages submitted via ``save_page()`` — populated by the
            tools during the run; the caller applies them afterwards.
    """

    gaps: GapBrokerProtocol
    source_id: str = "remediation"
    timeout: float = 120.0
    dry_run: bool = False
    pages: list[RemediationPageResult] = field(default_factory=list)


@dataclass
class OverviewDeps:
    """Runtime context for the agentic overview-synthesis agent.

    The agent lists pages via ``list_pages()``, reads full content via
    ``read_page()``, and submits the final overview via ``save_overview()``.

    Attributes:
        pages: Ordered list of page summaries (slug, title, content).
        source_id: Provenance / log label for the overview run.
        timeout: Per-call timeout in seconds.
        overview: The submitted overview text, populated by the
            ``save_overview`` tool.
    """

    pages: list[OverviewPage]
    source_id: str = "overview"
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
        claims: List of claim records (id + text).
        sources: List of source snippet records (slug + text).
        pages: List of wiki page records (slug + content).
        results: Collected verdicts, populated by the mark_* tools.
    """

    claims: list[VerificationClaim]
    sources: list[VerificationSource] = field(default_factory=list)
    pages: list[VerificationPage] = field(default_factory=list)
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
        chunks: List of chunk records (id, heading path, text).
        results: Collected context strings keyed by chunk_id.
        source_id: Identifier for the document being chunked.
        timeout: Per-call timeout in seconds.
    """

    doc_title: str
    doc_summary: str
    chunks: list[ContextChunk]
    results: dict[int, str] = field(default_factory=dict)
    source_id: str = "context_chunk"
    timeout: float = 120.0
    current_chunk: ContextChunk | None = None


@dataclass
class SummarizerDeps:
    """Runtime context for the agentic summarizer agent.

    The agent iterates over internal chunk nodes and writes a summary for each.

    Attributes:
        nodes: List of node records (id, heading path, content).
        results: Collected summaries keyed by node id.
        source_id: Identifier for the tree being summarized.
        timeout: Per-call timeout in seconds.
    """

    nodes: list[SummaryNode]
    results: dict[str, str] = field(default_factory=dict)
    source_id: str = "summarizer"
    timeout: float = 120.0
    current_node: SummaryNode | None = None


@dataclass
class DistillAgentDeps:
    """Runtime context for the agentic distill agent.

    The agent processes multiple tag groups, distilling each into a page.

    Attributes:
        groups: List of group records (tag, notes).
        pages: Collected page records keyed by group tag.
        source_id: Identifier for tracking.
        timeout: Per-call timeout in seconds.
    """

    groups: list[DistillGroup]
    pages: dict[str, DistillPage] = field(default_factory=dict)
    source_id: str = "distill"
    timeout: float = 120.0
    current_group: DistillGroup | None = None


@dataclass
class QueryAgentDeps:
    """Runtime context for the agentic query agent.

    The agent iterates over sub-questions and pages to synthesize an answer.

    Attributes:
        question: The original question to answer.
        sub_questions: List of pre-supplied sub-questions (usually empty —
            the system prompt has the agent decompose the question itself;
            ``results`` is keyed by whatever index ``add_finding`` uses).
        retriever: Raw wiki retrieval surface the tools call.
        max_sub_questions: Upper bound declared to the model in the prompt.
        tag_groups: Tag filters as an OR-of-ANDs (DNF); empty means no tag
            filter. Pushed down through the retriever by ``search_wiki``.
        attr_filters: Attribute filters as ``(key, value)`` pairs (``value``
            ``None`` = key-presence); all must hold. Pushed down likewise.
        results: Collected answer fragments keyed by sub-question index.
        source_id: Identifier for tracking.
        timeout: Per-call timeout in seconds.
    """

    question: str
    sub_questions: list[str]
    retriever: QuerySearchProtocol
    max_sub_questions: int = 5
    tag_groups: list[list[str]] = field(default_factory=list)
    attr_filters: list[tuple[str, str | None]] = field(default_factory=list)
    results: dict[int, str] = field(default_factory=dict)
    source_id: str = "query"
    timeout: float = 120.0
    current_sub_question: str | None = None


class VerificationClaim(TypedDict):
    """One claim the agentic verification agent may verify."""

    id: int
    text: str


class VerificationSource(TypedDict):
    """One cited source snippet offered as evidence."""

    slug: str
    text: str


class VerificationPage(TypedDict):
    """One wiki page the verifier may read in full."""

    slug: str
    content: str


class ContextChunk(TypedDict):
    """One leaf chunk the agentic context-chunk agent may situate.

    ``id`` is the identity of the live ``ChunkNode`` (``id(leaf)``), so the
    summarizer side and the result mapping agree on the key.
    """

    id: int
    heading_chain: list[str]
    text: str


class SummaryNode(TypedDict):
    """One internal chunk node the agentic summarizer may summarize."""

    id: str
    heading_chain: str
    content: str


class DistillNote(TypedDict):
    """One captured note inside a distill group."""

    title: str
    content: str
    source: str


class DistillGroup(TypedDict):
    """One tag group the agentic distill agent may distill."""

    tag: str
    notes: list[DistillNote]


class DistillPage(TypedDict, total=False):
    """A distilled page as returned by the model's ``save_page`` call.

    Untrusted model output: every key is optional and only ``title``,
    ``content`` and ``tags`` are read (defensively) by the caller.
    """

    title: str
    content: str
    tags: list[str]


class WikiHitProtocol(Protocol):
    """Minimal wiki-search hit surface.

    Structurally satisfied by :class:`kompend.search.retriever.SearchResult`.
    Declared locally so this leaf module stays free of ``kompend.search``
    imports (importing that package re-enters the agent import cycle).
    """

    article_id: str
    title: str
    content: str


class QuerySearchProtocol(Protocol):
    """Raw wiki retrieval used by the agentic query tools.

    Implementations return raw hits and raw page bodies; they must NOT
    synthesise answers — that would re-enter the query agent.
    """

    def search(
        self,
        query: str,
        top_k: int,
        *,
        tags_all: Sequence[str] | None = None,
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
    ) -> Sequence[WikiHitProtocol]: ...

    def read_page(self, slug: str) -> str | None: ...


class ConceptBrokerProtocol(Protocol):
    """Minimal broker surface used by the agentic generation tools.

    Declared locally (instead of importing the concrete
    :class:`kompend.compilation.concept_broker.ConceptBroker`) so this leaf
    module keeps zero ``kompend.compilation`` imports — importing any
    submodule of that package triggers ``compilation/__init__``, which
    pulls in ``agents.base`` and re-enters the import cycle.
    """

    def fetch(self, caller_id: str, limit: int = 5) -> list[dict[str, Any]]: ...


class GapBrokerProtocol(Protocol):
    """Minimal broker surface used by the agentic remediation tools.

    Mirrors :class:`ConceptBrokerProtocol`'s local-Protocol approach so
    ``agents.types`` stays a zero-``kompend.lint``-import leaf module.
    """

    def fetch(self, caller_id: str, limit: int = 5) -> list[dict[str, Any]]: ...

    def mark_done(self, name: str) -> None: ...

    def mark_unfixable(self, name: str, reason: str) -> None: ...
