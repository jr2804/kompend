"""Pydantic config models for KNOX.

Single source of truth for the config schema: each field carries its type,
default, and optional env-var alias in one declaration. ``DEFAULTS`` and
``_ENV_MAP`` in ``_core.py`` are auto-generated from these models.

The six sub-configs used by ``PipelineConfig`` (``LLMConfig``, ``BudgetConfig``,
``OutputConfig``, ``EmbeddingConfig``, ``PdfConfig``, ``OrganizeConfig``) double
as runtime types — they are constructed in ``from_resolved()`` and held as
fields on ``PipelineConfig``.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from knox.core.config.keys import MANDATORY_KINDS, EnvVar
from knox.core.paths import CACHE_DIR, KNOX_DIR


class ExportMode(StrEnum):
    """Where extracted data goes in a ``sources export``.

    - ``OFF``: skip entirely
    - ``INLINE``: markdown section in the exported file
    - ``ASSET``: JSON file in the export's assets directory
    """

    OFF = "off"
    INLINE = "inline"
    ASSET = "asset"


class ExtractionStrategy(StrEnum):
    """How per-chunk extractions are combined into one document-level Extraction.

    - ``reduce``: map-reduce — extract each batch independently, then a
      second agent pass merges partial results into a coherent
      document-level Extraction.
    - ``incremental``: rolling — feed chunks sequentially into one agent
      conversation via ``message_history``, so each batch sees the
      accumulated context from prior batches.
    - ``agentic``: stateful — one agent per source walks sections in
      order, accumulating a structured ``ExtractionMemory`` (concepts,
      entities, tags) with compaction-managed digest injection between
      sections. Best coherence for long/high-value sources; sequential
      within a source, not cacheable per-batch.
    """

    REDUCE = "reduce"
    INCREMENTAL = "incremental"
    AGENTIC = "agentic"


# ── Enums ────────────────────────────────────────────────────────────────


class ImageMode(StrEnum):
    """Image handling mode for document parsing.

    - ``FILE``: save images as files (default)
    - ``BASE64``: embed images as base64-encoded data URIs
    """

    FILE = "file"
    BASE64 = "base64"


class LlmDataFormat(StrEnum):
    """Single source of truth for structured data serialization across all LLM communication.

    Covers two distinct use cases that share the same compact formats:

    1. **LLM input** — structured data serialized before injection into LLM prompts
       (e.g. concept batches in generation, probes in WICER). These use TOON/ISON
       to reduce prompt token count by 30-70%, leaving more context window for
       model output and reducing truncation-related malformed responses from
       small local models.

    2. **Downstream LLM ingestion** — structured data serialized by one agent and
       consumed by another (e.g. query command output fed to an agent consumer).
       Even though this is technically "output", the structured encoding is
       the *input* for subsequent LLM calls, so it benefits from the same
       token efficiency.

    Formats
    -------
    - ``JSON``: Standard JSON (default). Uses API-level ``response_format``
      guarantee. Always the LLM *output* format — providers enforce valid JSON
      at the protocol level via ``response_format: {"type": "json_object"}``.
      TOON/ISON are not yet widely understood by models and are counter-
      productive as output formats.
    - ``TOON``: Token-Oriented Object Notation. 30-60% fewer tokens vs JSON.
    - ``ISON``: Interchange Simple Object Notation. 30-70% fewer tokens vs JSON.
    - ``MARKDOWN``: Human-readable prose (formerly ``TEXT``). Alias for
      ``"text"`` — used when the consumer is a human reader, not an LLM.

    Note: ``response_format`` (JSON-enforced output) is orthogonal to this
    enum. The API-level ``response_format`` always produces JSON regardless
    of which ``LlmDataFormat`` is configured for prompt input. This enum
    controls *serialization of structured data before injection*, not the
    format of the LLM's response.
    """

    JSON = "json"
    TOON = "toon"
    ISON = "ison"
    MARKDOWN = "text"

    # Legacy alias — kept for backward compat with old "TEXT" references
    TEXT = "text"


class OutputFormat(StrEnum):
    """Output format for the compiled wiki.

    OKF (Open Knowledge Format v0.2) is a **compliance spec**, not a format
    — the default markdown output is OKF-compliant by default.

    - ``MARKDOWN``: Markdown vault (OKF v0.2-compliant, default)
    - ``JSON``: JSON file
    - ``TYPST``: Typst file
    """

    MARKDOWN = "markdown"
    JSON = "json"
    TYPST = "typst"


class PdfOcrMode(StrEnum):
    """Whether Docling runs its OCR pass on a PDF.

    - ``AUTO``: run OCR only when the document has no usable text layer
      (scanned/image-only PDFs). Born-digital PDFs skip the pass entirely,
      which is both faster and content-identical because OCR adds nothing
      when every sampled page already carries extractable text.
    - ``ALWAYS``: force OCR even on born-digital PDFs.
    - ``NEVER``: never run OCR.
    """

    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"


class PdfTableMode(StrEnum):
    """TableFormer recognition mode for PDF tables (Docling ``TableStructureOptions``).

    - ``ACCURATE``: cell-matching on — slower, highest table fidelity (default).
    - ``FAST``: cell-matching off — faster, but loses some cell detail and can
      drop text inside detected table regions.
    """

    ACCURATE = "accurate"
    FAST = "fast"


class Verbosity(StrEnum):
    """Synthesis verbosity for the ``query`` command.

    Drives both the synthesis system prompt wording and the
    ``max_tokens`` budget for the synthesis LLM call. Defaults to
    ``standard``; configurable via ``search.synthesis.verbosity``.
    """

    BRIEF = "brief"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class _KnoxBaseModel(BaseModel):
    """Base for all config models.

    ``populate_by_name=True`` allows construction via field name (e.g.
    ``LLMConfig(default_model="x")``) even when a ``validation_alias`` is set
    (e.g. ``"KNOX_LLM_DEFAULT_MODEL"``).
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


# Env-var mapping: every field below carries its ``KNOX_*`` env override as a
# ``validation_alias``; ``_build_env_map`` in ``engine/_schema.py`` turns those
# aliases into the config-map automatically.

# ── Compile sub-configs ──────────────────────────────────────────────────


class RlmConfig(_KnoxBaseModel):
    """RLM (recursive research-loop) agent configuration (``[compile.llm.rlm]``).

    ``enable`` turns the capability on; the remaining fields set the
    execution budget directly.
    """

    enable: bool = False
    max_tool_calls: int = 50
    code_timeout: float = 60.0
    sub_model: str | None = None
    max_depth: int = 2
    node_budget: int | None = None


class LLMConfig(_KnoxBaseModel):
    """Configuration for LLM models, timeouts, and fallback behavior."""

    default_model: str | None = Field(default=None, validation_alias=EnvVar.LLM_DEFAULT_MODEL)
    vlm_model: str | None = Field(default=None, validation_alias=EnvVar.VLM_MODEL)
    fallback_chain: list[str] = Field(default_factory=list, validation_alias=EnvVar.LLM_FALLBACK_CHAIN)
    fallback_max_retries: int = Field(default=2, validation_alias=EnvVar.LLM_FALLBACK_MAX_RETRIES)
    timeout: float = Field(default=120.0, validation_alias=EnvVar.LLM_TIMEOUT)
    fallback_max_input: int = Field(default=32768, validation_alias=EnvVar.LLM_FALLBACK_MAX_INPUT)
    fallback_max_output: int = Field(default=4096, validation_alias=EnvVar.LLM_FALLBACK_MAX_OUTPUT)
    num_ctx_min: int = Field(default=2048, validation_alias=EnvVar.LLM_NUM_CTX_MIN)
    num_ctx_max: int = Field(default=131072, validation_alias=EnvVar.LLM_NUM_CTX_MAX)
    context_window_cap: int | None = Field(default=None, validation_alias=EnvVar.LLM_CONTEXT_WINDOW_CAP, description="DEPRECATED — use context_window")
    context_window: int | None = Field(
        default=None, validation_alias=EnvVar.LLM_CONTEXT_WINDOW, description="Context window length override/cap applied to every discovered model"
    )
    data_format: LlmDataFormat = Field(default=LlmDataFormat.JSON, validation_alias=EnvVar.LLM_DATA_FORMAT)
    output_retries: int = Field(default=2, validation_alias=EnvVar.LLM_OUTPUT_RETRIES)
    output_retries_enabled: bool = Field(default=True, validation_alias=EnvVar.LLM_OUTPUT_RETRIES_ENABLED)
    #: Per-turn output-validation retry budget for tool-based (agentic) agents.
    #: Defaults higher than ``output_retries`` because agentic agents run
    #: multi-turn loops (fetch concepts → read segment → generate) where each
    #: turn can be invalidated; 2 retries is too tight for the typical
    #: small-model interaction.  Does NOT govern tool-call retries — those
    #: share the legacy ``output_retries`` budget via pydantic-ai's
    #: ``AgentRetries`` TypedDict (tool failures 4× in a row are a real bug;
    #: output validation 4× in a row is a small-model normal occurrence).
    agentic_output_retries: int = Field(default=4, validation_alias=EnvVar.LLM_AGENTIC_OUTPUT_RETRIES)
    #: Rolling window (in output tokens) for the short-term tok/s rate.
    throughput_window_tokens: int = Field(default=1000, validation_alias=EnvVar.LLM_THROUGHPUT_WINDOW_TOKENS)
    delta_retry_enabled: bool = Field(default=True, validation_alias=EnvVar.LLM_DELTA_RETRY_ENABLED)
    batch_failure_threshold_ratio: float = Field(default=0.5, validation_alias=EnvVar.LLM_BATCH_FAILURE_THRESHOLD_RATIO)
    batch_failure_threshold_min: int = Field(default=5, validation_alias=EnvVar.LLM_BATCH_FAILURE_THRESHOLD_MIN)
    reasoning_effort: str | None = Field(default=None, validation_alias=EnvVar.LLM_REASONING_EFFORT)
    context_manager: str | bool = Field(default="default", validation_alias=EnvVar.LLM_CONTEXT_MANAGER)
    #: RLM recursive research-loop configuration (see :class:`RlmConfig`);
    #: ``None`` disables. Only the group form is accepted — no scalar
    #: shorthand.
    rlm: RlmConfig | None = None
    extraction_strategy: ExtractionStrategy = Field(
        default=ExtractionStrategy.INCREMENTAL,
        validation_alias=EnvVar.LLM_EXTRACTION_STRATEGY,
    )
    slug_max_length: int = Field(default=80, validation_alias=EnvVar.LLM_SLUG_MAX_LENGTH)
    batch_retry_attempts: int = Field(default=3, validation_alias=EnvVar.LLM_BATCH_RETRY_ATTEMPTS)
    min_segment_tokens: int = Field(default=512, validation_alias=EnvVar.LLM_MIN_SEGMENT_TOKENS)
    default_chars_per_token: float = Field(default=3.85, validation_alias=EnvVar.LLM_DEFAULT_CHARS_PER_TOKEN)


class BudgetConfig(_KnoxBaseModel):
    """Token budgets and structural limits for LLM operations.

    Sizing strategy:
    - Input chunks: ``clamp(context_window - reserve, chunk_size_min, chunk_size_max)`` (knox-7sp)
    - LLM output reserves: per-phase absolute ``*_output_max`` (knox-vxd).  Absolute
      reserves scale correctly to large context windows; an earlier ratio-based
      design (``*_output_ratio`` / ``*_output_min``) was specified but never
      implemented and has been removed.
    """

    verify_min_ngram: int = 3
    wicer_max_iterations: int = Field(default=3, validation_alias=EnvVar.BUDGET_WICER_MAX_ITERATIONS)
    min_content_length: int = Field(default=50, validation_alias=EnvVar.BUDGET_MIN_CONTENT_LENGTH)
    min_concept_sources: int = Field(default=2, validation_alias=EnvVar.BUDGET_MIN_CONCEPT_SOURCES)
    max_concept_pages: int = Field(default=50, validation_alias=EnvVar.BUDGET_MAX_CONCEPT_PAGES)
    chunk_size_min: int = Field(default=2048, validation_alias=EnvVar.BUDGET_CHUNK_SIZE_MIN)
    chunk_size_max: int = Field(default=131072, validation_alias=EnvVar.BUDGET_CHUNK_SIZE_MAX)
    extract_output_max: int = Field(default=8192, validation_alias=EnvVar.BUDGET_EXTRACT_OUTPUT_MAX)
    gen_output_max: int = Field(default=16384, validation_alias=EnvVar.BUDGET_GEN_OUTPUT_MAX)
    overview_output_max: int = Field(default=4096, validation_alias=EnvVar.BUDGET_OVERVIEW_OUTPUT_MAX)
    verify_output_max: int = 4096
    lint_output_max: int = 8192
    wicer_probe_output_max: int = 4096
    wicer_eval_output_max: int = 4096
    summary_max_content_chars: int = 8000
    summary_target_chars: int = 600
    summary_skip_single_child: bool = True
    summary_chunk_threshold: int = 10
    capture_source_char_max: int = Field(default=4000, validation_alias=EnvVar.BUDGET_CAPTURE_SOURCE_CHAR_MAX)
    concept_merge_threshold: float = Field(default=0.82, validation_alias=EnvVar.BUDGET_CONCEPT_MERGE_THRESHOLD)
    concept_string_merge_threshold: float = Field(default=0.70, validation_alias=EnvVar.BUDGET_CONCEPT_STRING_MERGE_THRESHOLD)
    jaccard_min_token_len: int = Field(default=1, validation_alias=EnvVar.BUDGET_JACCARD_MIN_TOKEN_LEN)
    edge_min_title_len: int = Field(default=4, validation_alias=EnvVar.BUDGET_EDGE_MIN_TITLE_LEN)


class OutputConfig(_KnoxBaseModel):
    """Configuration for output formatting and rendering.

    ``format`` defaults to ``markdown`` (an OKF v0.2-compliant vault).
    """

    format: OutputFormat = Field(default=OutputFormat.MARKDOWN, validation_alias=EnvVar.FORMAT)
    flavor: str | None = Field(default=None, validation_alias=EnvVar.OUTPUT_FLAVOR)
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    rumdl_config: Path | None = Field(default=None, validation_alias=EnvVar.RUMDL_CONFIG)
    max_related_links: int = Field(default=15, validation_alias=EnvVar.OUTPUT_MAX_RELATED_LINKS)


class EmbeddingConfig(_KnoxBaseModel):
    """Configuration for embedding models.

    Vector dimensionality is detected from the model at runtime via the
    embedding provider — no ``dim`` config value is needed.
    """

    model: str = Field(default="")
    batch_size: int = Field(default=10, validation_alias=EnvVar.EMBEDDING_BATCH_SIZE)


class PdfConfig(_KnoxBaseModel):
    """Configuration for PDF parsing.

    *chunking* keeps per-doc memory bounded on large PDFs by splitting above
    ``max_pages_before_chunking`` into sequential page-range segments.

    *enrichment* flags turn on Docling's recovery passes. ``table_structure``
    and ``formula_enrichment`` default ``True``: dropping them silently loses
    tables or equations from every parsed PDF and the resulting wiki is what
    knox compiled.

    ``code_enrichment`` defaults ``False`` — it is the one pass whose native
    representation is already present. Docling's CodeFormulaV2 VLM re-OCRs
    every ``CodeItem``, but a born-digital PDF already carries correct code in
    its text layer, and the pass costs ~140 s per code block on CPU (measured:
    9 code blocks added ~1250 s to a 16-page range, i.e. >30 min for a
    code-heavy manual). Disabling it changes nothing for text-layer code; enable
    it only when code renders as images that OCR handles poorly.

    ``ocr`` is not an enrichment flag: OCR only recovers text a born-digital
    PDF already carries in its text layer, so ``AUTO`` skips it there and
    reserves it for scans. ``num_threads`` (0 = all cores) and ``table_mode``
    are the CPU-throughput knobs; Docling otherwise defaults to 4 threads.
    """

    max_pages_before_chunking: int = Field(default=50, validation_alias=EnvVar.PDF_MAX_PAGES_BEFORE_CHUNKING)
    page_chunk_size: int = Field(default=30, validation_alias=EnvVar.PDF_PAGE_CHUNK_SIZE)
    table_structure: bool = Field(default=True, validation_alias=EnvVar.PDF_TABLE_STRUCTURE)
    table_mode: PdfTableMode = Field(default=PdfTableMode.ACCURATE, validation_alias=EnvVar.PDF_TABLE_MODE)
    formula_enrichment: bool = Field(default=True, validation_alias=EnvVar.PDF_FORMULA_ENRICHMENT)
    code_enrichment: bool = Field(default=False, validation_alias=EnvVar.PDF_CODE_ENRICHMENT)
    ocr: PdfOcrMode = Field(default=PdfOcrMode.AUTO, validation_alias=EnvVar.PDF_OCR)
    ocr_min_text_chars_per_page: int = Field(
        default=50,
        ge=1,
        validation_alias=EnvVar.PDF_OCR_MIN_TEXT_CHARS_PER_PAGE,
    )
    num_threads: int = Field(default=0, ge=0, validation_alias=EnvVar.PDF_NUM_THREADS)


# ── Other section configs ────────────────────────────────────────────────


class PathsConfig(_KnoxBaseModel):
    """Vault path overrides (all optional — resolved from base_dir at runtime)."""

    wiki_dir: str | None = Field(default=None, validation_alias=EnvVar.WIKI_DIR)
    sources_dir: str = Field(default="sources", validation_alias=EnvVar.SOURCES_DIR)
    output_dir: str = Field(default="wiki", validation_alias=EnvVar.OUTPUT_DIR)
    cache_dir: str = Field(default=CACHE_DIR, validation_alias=EnvVar.CACHE_DIR)
    state_dir: str = Field(default=KNOX_DIR, validation_alias=EnvVar.STATE_DIR)


class LintConfig(_KnoxBaseModel):
    """Lint configuration. ``enabled=None`` means all structural rules."""

    enabled: list[str] | None = None
    fuzzy_threshold: float = Field(default=0.85, validation_alias=EnvVar.LINT_FUZZY_THRESHOLD)
    model: str | None = Field(
        default=None,
        validation_alias=EnvVar.LINT_MODEL,
        description="Model override for LLM-powered lint steps (factual-consistency, clarity-score, ...). Falls back to compile.llm.default_model when unset.",
    )


class EvalConfig(_KnoxBaseModel):
    """Evaluation configuration."""

    threshold: float | None = Field(default=None, validation_alias=EnvVar.EVAL_THRESHOLD)
    history_size: int = 20


class DedupConfig(_KnoxBaseModel):
    """Deduplication configuration (opt-in by default)."""

    enabled: bool = Field(default=False, validation_alias=EnvVar.DEDUP_ENABLED)
    matchers: str = Field(default="embedding,minhash", validation_alias=EnvVar.DEDUP_MATCHERS)
    threshold_identical: float = Field(default=0.95, validation_alias=EnvVar.DEDUP_THRESHOLD_IDENTICAL)
    threshold_similar: float = Field(default=0.80, validation_alias=EnvVar.DEDUP_THRESHOLD_SIMILAR)
    model: str = Field(default="", validation_alias=EnvVar.DEDUP_MODEL)
    minhash_lsh_threshold: float = Field(default=0.8, validation_alias=EnvVar.DEDUP_MINHASH_LSH_THRESHOLD)


class TrustConfig(_KnoxBaseModel):
    """Privacy and trust configuration."""

    privacy_filter: bool = Field(default=True)
    user_tracking: bool = Field(default=True, validation_alias=EnvVar.USER_TRACKING)


class QueryConfig(_KnoxBaseModel):
    """Query command configuration."""

    format: LlmDataFormat = LlmDataFormat.MARKDOWN


class SearchExpansionConfig(_KnoxBaseModel):
    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_EXPANSION_ENABLED)
    variants: int = Field(default=3, validation_alias=EnvVar.SEARCH_EXPANSION_VARIANTS)


class SearchCragConfig(_KnoxBaseModel):
    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_CRAG_ENABLED)
    threshold: float = Field(default=0.3, validation_alias=EnvVar.SEARCH_CRAG_THRESHOLD)


class SearchRerankerConfig(_KnoxBaseModel):
    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_RERANKER_ENABLED)
    model: str = Field(default="", validation_alias=EnvVar.SEARCH_RERANKER_MODEL)
    top_k: int = Field(default=20, validation_alias=EnvVar.SEARCH_RERANKER_TOP_K)


class SearchHydeConfig(_KnoxBaseModel):
    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_HYDE_ENABLED)


class SearchContextualConfig(_KnoxBaseModel):
    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_CONTEXTUAL_ENABLED)
    model: str = Field(default="", validation_alias=EnvVar.SEARCH_CONTEXTUAL_MODEL)


class SearchAgenticConfig(_KnoxBaseModel):
    """Opt-in agentic query path (agent searches the wiki via tools).

    Default off — the existing retrieve-then-synthesize path is unchanged.
    """

    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_AGENTIC_ENABLED)
    model: str = Field(default="", validation_alias=EnvVar.SEARCH_AGENTIC_MODEL)
    max_sub_questions: int = Field(default=5, validation_alias=EnvVar.SEARCH_AGENTIC_MAX_SUB_QUESTIONS)


class SearchSynthesisTokensConfig(_KnoxBaseModel):
    brief: int = Field(default=300, validation_alias=EnvVar.SEARCH_SYNTHESIS_TOKENS_BRIEF)
    standard: int = Field(default=800, validation_alias=EnvVar.SEARCH_SYNTHESIS_TOKENS_STANDARD)
    comprehensive: int = Field(default=2000, validation_alias=EnvVar.SEARCH_SYNTHESIS_TOKENS_COMPREHENSIVE)


class SearchSynthesisConfig(_KnoxBaseModel):
    verbosity: Verbosity = Field(default=Verbosity.STANDARD, validation_alias=EnvVar.SEARCH_SYNTHESIS_VERBOSITY)
    tokens: SearchSynthesisTokensConfig = Field(default_factory=SearchSynthesisTokensConfig)


class AttachmentsConfig(_KnoxBaseModel):
    """Attachment lifecycle configuration (auto-adoption, vector indexing)."""

    ingest_adopted: bool = Field(
        default=False,
        validation_alias=EnvVar.ATTACHMENTS_INGEST_ADOPTED,
        description=(
            "Global gate for attachment ingestion: auto-adopted attachments "
            "(files under attachments/<slug>/ without a manifest entry) are "
            "registered but NOT ingestible by default. When True, adopted "
            "text/image files become ingestible. Effective ingestibility at "
            "compile is manifest flag AND this gate."
        ),
    )
    embed: bool = Field(
        default=False,
        validation_alias=EnvVar.ATTACHMENTS_EMBED,
        description="Whether ingestible attachments (text/image captions) are included in the vector index. Replaces the deprecated search.embed_attachments.",
    )


class SearchConfig(_KnoxBaseModel):
    expansion: SearchExpansionConfig = Field(default_factory=SearchExpansionConfig)
    crag: SearchCragConfig = Field(default_factory=SearchCragConfig)
    reranker: SearchRerankerConfig = Field(default_factory=SearchRerankerConfig)
    hyde: SearchHydeConfig = Field(default_factory=SearchHydeConfig)
    contextual: SearchContextualConfig = Field(default_factory=SearchContextualConfig)
    agentic: SearchAgenticConfig = Field(default_factory=SearchAgenticConfig)
    synthesis: SearchSynthesisConfig = Field(default_factory=SearchSynthesisConfig)
    embed_attachments: bool = Field(
        default=False,
        description="DEPRECATED — use attachments.embed. Whether ingestible attachments are included in the vector index.",
    )


class OrganizeConfig(_KnoxBaseModel):
    """Configuration for ``knox organize`` LLM-based file categorization."""

    max_depth: int = Field(default=3, validation_alias=EnvVar.ORGANIZE_MAX_DEPTH)
    min_subcategories: int = Field(default=2, validation_alias=EnvVar.ORGANIZE_MIN_SUBCATEGORIES)
    min_files_per_category: int = 2
    collapse_single_file: bool = Field(default=True, validation_alias=EnvVar.ORGANIZE_COLLAPSE_SINGLE_FILE)


class IgnoreConfig(_KnoxBaseModel):
    patterns: list[str] = Field(default_factory=list)


class DisplayConfig(_KnoxBaseModel):
    """Display/preview truncation limits.

    Each field caps how many items or how much text a single rendered
    section (index, log report, related-titles list, etc.) shows.
    Limits are deliberately configurable so users with long source lists
    or large captured notes can widen them without touching code.
    """

    source_names: int = Field(default=20, validation_alias=EnvVar.DISPLAY_SOURCE_NAMES)
    related_titles: int = Field(default=15, validation_alias=EnvVar.DISPLAY_RELATED_TITLES)
    doc_headings: int = Field(default=5, validation_alias=EnvVar.DISPLAY_DOC_HEADINGS)
    concept_names: int = Field(default=3, validation_alias=EnvVar.DISPLAY_CONCEPT_NAMES)
    memory_entries: int = Field(default=10, validation_alias=EnvVar.DISPLAY_MEMORY_ENTRIES)
    excerpt_chars: int = Field(default=300, validation_alias=EnvVar.DISPLAY_EXCERPT_CHARS)
    error_chars: int = Field(default=200, validation_alias=EnvVar.DISPLAY_ERROR_CHARS)
    json_chars: int = Field(default=300, validation_alias=EnvVar.DISPLAY_JSON_CHARS)
    cli_element_chars: int = Field(default=80, validation_alias=EnvVar.DISPLAY_CLI_ELEMENT_CHARS)
    tier_sample_size: int = Field(default=5, validation_alias=EnvVar.DISPLAY_TIER_SAMPLE_SIZE)
    cli_arg_preview: int = Field(default=4, validation_alias=EnvVar.DISPLAY_CLI_ARG_PREVIEW)
    log_error_chars: int = Field(default=500, validation_alias=EnvVar.DISPLAY_LOG_ERROR_CHARS)
    lint_feedback_chars: int = Field(default=200, validation_alias=EnvVar.DISPLAY_LINT_FEEDBACK_CHARS)
    memory_query_chars: int = Field(default=50, validation_alias=EnvVar.DISPLAY_MEMORY_QUERY_CHARS)
    mcp_element_chars: int = Field(default=500, validation_alias=EnvVar.DISPLAY_MCP_ELEMENT_CHARS)
    wiki_topics: int = Field(default=30, validation_alias=EnvVar.DISPLAY_WIKI_TOPICS)
    query_expansions: int = Field(default=7, validation_alias=EnvVar.DISPLAY_QUERY_EXPANSIONS)
    web_nodes: int = Field(default=10, validation_alias=EnvVar.DISPLAY_WEB_NODES)
    search_snippet_chars: int = Field(default=300, validation_alias=EnvVar.DISPLAY_SEARCH_SNIPPET_CHARS)


class NotesConfig(_KnoxBaseModel):
    file: str = Field(default="NOTES.md", validation_alias=EnvVar.NOTES_FILE)


class IngestConfig(_KnoxBaseModel):
    parser: str = Field(default="liteparse", validation_alias=EnvVar.INGEST_PARSER)
    parser_fallback: bool = Field(
        default=True,
        validation_alias=EnvVar.INGEST_PARSER_FALLBACK,
        description=(
            "Parse-tier chain fallback *between configured backends* — a backend that fails "
            "hands off to the next backend in `[ingest] parser` that claims the file. "
            "Distinct from the *text* fallback (pandoc filling in at the end of the chain "
            "for unclaimed plain-text files)."
        ),
    )


class DownloaderConfig(_KnoxBaseModel):
    undetected: bool = Field(default=True, validation_alias=EnvVar.DOWNLOADER_UNDETECTED)


class ExportConfig(_KnoxBaseModel):
    """Where extracted knowledge lands in ``knox sources export``."""

    annotations: ExportMode = Field(default=ExportMode.INLINE, validation_alias=EnvVar.EXPORT_ANNOTATIONS)
    summary: ExportMode = Field(default=ExportMode.INLINE, validation_alias=EnvVar.EXPORT_SUMMARY)


class ExtractionConfig(_KnoxBaseModel):
    """Category extraction configuration.

    ``categories`` maps extra page-kind names to prompt instructions. The
    mandatory taxonomy (``concept``, ``entity``, ``summary``, ``comparison``)
    is always active and cannot be shadowed.
    """

    categories: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _no_reserved_kinds(self) -> ExtractionConfig:
        reserved = sorted(MANDATORY_KINDS & set(self.categories))
        if reserved:
            raise ValueError("compile.extraction.categories may not redefine mandatory kinds: " + ", ".join(reserved))
        for name, desc in self.categories.items():
            if not desc.strip():
                raise ValueError(f"compile.extraction.categories[{name!r}] needs a non-empty instruction")
        return self


class ProviderModelConfig(_KnoxBaseModel):
    """Per-model settings within a provider (``[providers.<name>.models."<model>"]``)."""

    context_window: int | None = Field(
        default=None, description="Context window length for this model (finest granularity, wins over provider-level and catalog values)"
    )


class ProviderEndpoint(_KnoxBaseModel):
    """A user-defined provider endpoint.

    Defined under ``[providers.<name>]`` in ``knox.toml`` (recommended:
    globally in ``~/.config/knox/config.toml``, since providers carry
    credentials):

    .. code-block:: toml

        [providers.my-llama]
        base_url = "http://localhost:8081/v1"
        api_key = "${MY_API_KEY}"          # optional; env var reference
        api_type = "openai-chat"           # openai-chat | openai-responses | anthropic
        context_window = 32768             # default context window for all its models

        [providers.my-llama.models."mistral-nemo"]
        context_window = 131072            # per-model override (finest granularity)

    The provider name (``my-llama``) becomes the ``provider`` part in
    ``provider/model`` strings, e.g. ``my-llama/mistral-nemo``.

    A section may omit ``base_url`` entirely to act as an **override block**
    for an integrated provider (e.g. ``[providers.ollama]`` with only
    ``max_connections``) — settings apply, model building still routes
    through the native/catalog path.
    """

    base_url: str | None = Field(
        default=None,
        description="Provider base URL (e.g. http://localhost:8081/v1). None = override-only block for an integrated provider",
    )
    api_key: str | None = Field(default=None, description="API key or env var reference like ${VAR_NAME}")
    api_type: Literal["openai-chat", "openai-responses", "anthropic"] = Field(
        default="openai-chat",
        description="Wire protocol the endpoint speaks: openai-chat (/v1/chat/completions), openai-responses (/v1/responses), or anthropic (/v1/messages)",
    )
    context_window: int | None = Field(
        default=None, description="Default context window length for all models on this provider (overrides auto-discovery; per-model entries win)"
    )
    context_window_cap: int | None = Field(
        default=None, description="DEPRECATED — use context_window. Old per-provider override key, still honored when context_window is unset"
    )
    models: dict[str, ProviderModelConfig] = Field(default_factory=dict, description="Per-model settings keyed by model name")
    max_connections: int | None = Field(
        default=None,
        description="Max parallel LLM connections to this provider (client-side). Clips compile.workers when set. None = no limit",
    )


class SkillConfig(_KnoxBaseModel):
    """A named skill — user-defined instructions injected into the PydanticAI agent's system prompt.

    Skills are defined under ``[compile.skills.<name>]`` in config:

    .. code-block:: toml

        [compile.skills."domain-terms"]
        instructions = '''...'''  # free-form skill instructions
    """

    instructions: str


# ── Compile section (top-level flags + nested sub-configs) ──────────────


class CompileConfig(_KnoxBaseModel):
    """Compile section: top-level flags and nested sub-configs."""

    # Hidden duplicates of paths.* (kept for backward compat, stripped from scaffold)
    sources_dir: str = "sources"
    output_dir: str = "wiki"

    compiler: str = "graph"
    image_mode: ImageMode = Field(default=ImageMode.FILE, validation_alias=EnvVar.IMAGE_MODE)
    vlm_min_image_bytes: int = 2048
    vlm_min_image_dimension: int = 64
    mode: str = "full"
    workers: int = Field(default=1, validation_alias=EnvVar.WORKERS)
    no_cache: bool = Field(default=False, validation_alias=EnvVar.NO_CACHE)
    force: bool = Field(default=False, validation_alias=EnvVar.FORCE)
    in_place_sources: bool = Field(default=False)
    memory_enabled: bool = Field(default=True)
    memory_half_life_days: float = Field(default=90.0, validation_alias=EnvVar.MEMORY_HALF_LIFE_DAYS)
    memory_decay_threshold: float = Field(default=0.3, validation_alias=EnvVar.MEMORY_DECAY_THRESHOLD)
    review: bool = False
    bypass_review: bool = False
    provenance: bool = False
    private: bool = False
    verify: bool = True
    synthesize: bool = True
    index: bool = True
    lint: bool = Field(default=False)
    start_at_tier: str = Field(default="index", validation_alias=EnvVar.START_AT_TIER)
    stop_at_tier: str = Field(default="output", validation_alias=EnvVar.STOP_AT_TIER)
    respect_gitignore: bool = True
    source_tags: list[str] = Field(default_factory=list)
    tier_state_file: str | None = None

    # Nested sub-configs
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    pdf: PdfConfig = Field(
        default_factory=PdfConfig,
        description=(
            "PDF parsing knobs. ``max_pages_before_chunking`` bounds memory "
            "on large PDFs; ``table_structure``/``formula_enrichment`` default "
            "to true and ``code_enrichment`` to false (see PdfConfig). "
            "``ocr`` defaults to ``auto``; ``num_threads``/``table_mode`` "
            "control CPU throughput. "
            "See PdfConfig."
        ),
    )
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    skills: dict[str, SkillConfig] = Field(default_factory=dict)


# ── Root config ──────────────────────────────────────────────────────────


class KnoxConfig(_KnoxBaseModel):
    """Root config model — single source of truth for DEFAULTS and _ENV_MAP.

    ``DEFAULTS = KnoxConfig().model_dump()`` and ``_ENV_MAP`` is auto-generated
    by walking the model's ``validation_alias`` fields.
    """

    display: DisplayConfig = Field(default_factory=DisplayConfig)
    compile: CompileConfig = Field(default_factory=CompileConfig)
    lint: LintConfig = Field(default_factory=LintConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    trust: TrustConfig = Field(default_factory=TrustConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    attachments: AttachmentsConfig = Field(default_factory=AttachmentsConfig)
    organize: OrganizeConfig = Field(default_factory=OrganizeConfig)
    ignore: IgnoreConfig = Field(default_factory=IgnoreConfig)
    hooks: dict[str, Any] = Field(default_factory=dict)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    notes: NotesConfig = Field(default_factory=NotesConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    downloader: DownloaderConfig = Field(default_factory=DownloaderConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    providers: dict[str, ProviderEndpoint] = Field(default_factory=dict)
