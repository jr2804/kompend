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
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from knox.core.paths import CACHE_DIR, KNOX_DIR


class _KnoxBaseModel(BaseModel):
    """Base for all config models.

    ``populate_by_name=True`` allows construction via field name (e.g.
    ``LLMConfig(default_model="x")``) even when a ``validation_alias`` is set
    (e.g. ``"KNOX_LLM_DEFAULT_MODEL"``).
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


# ── Enums ────────────────────────────────────────────────────────────────


class ImageMode(StrEnum):
    """Image handling mode for document parsing.

    - ``FILE``: save images as files (default)
    - ``BASE64``: embed images as base64-encoded data URIs
    """

    FILE = "file"
    BASE64 = "base64"


class OutputFormat(StrEnum):
    """Output format for the compiled wiki.

    - ``OKF``: Open Knowledge Format v0.1 bundle (default)
    - ``MARKDOWN``: Markdown files (Obsidian vault mode, legacy)
    - ``JSON``: JSON file
    - ``TYPST``: Typst file
    """

    OKF = "okf"
    MARKDOWN = "markdown"
    JSON = "json"
    TYPST = "typst"


class LlmDataFormat(StrEnum):
    """Data format for LLM communication (prompt input and response output).

    - ``JSON``: Standard JSON (default). Uses API-level response_format guarantee.
    - ``TOON``: Token-Oriented Object Notation. 30-60% fewer tokens vs JSON.
    - ``ISON``: Interchange Simple Object Notation. 30-70% fewer tokens vs JSON.
    """

    JSON = "json"
    TOON = "toon"
    ISON = "ison"


class QueryFormat(StrEnum):
    """Output format for the ``query`` command.

    ``text`` is human-readable prose. ``json``/``toon``/``ison`` are
    structured encodings of ``{answer, citations}`` for LLM/agent
    consumers; ``toon``/``ison`` reuse :class:`LlmDataFormat` encoders
    for token efficiency. Default configurable via ``query.format``.
    """

    TEXT = "text"
    JSON = "json"
    TOON = "toon"
    ISON = "ison"


class Verbosity(StrEnum):
    """Synthesis verbosity for the ``query`` command.

    Drives both the synthesis system prompt wording and the
    ``max_tokens`` budget for the synthesis LLM call. Defaults to
    ``standard``; configurable via ``search.synthesis.verbosity``.
    """

    BRIEF = "brief"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


# ── Compile sub-configs ──────────────────────────────────────────────────


class LLMConfig(_KnoxBaseModel):
    """Configuration for LLM models, timeouts, and fallback behavior."""

    default_model: str | None = Field(default=None, validation_alias="KNOX_LLM_DEFAULT_MODEL")
    vlm_model: str | None = Field(default=None, validation_alias="KNOX_VLM_MODEL")
    fallback_chain: list[str] = Field(default_factory=list, validation_alias="KNOX_LLM_FALLBACK_CHAIN")
    fallback_max_retries: int = Field(default=2, validation_alias="KNOX_LLM_FALLBACK_MAX_RETRIES")
    timeout: float = Field(default=120.0, validation_alias="KNOX_LLM_TIMEOUT")
    fallback_max_input: int = Field(default=32768, validation_alias="KNOX_LLM_FALLBACK_MAX_INPUT")
    fallback_max_output: int = Field(default=4096, validation_alias="KNOX_LLM_FALLBACK_MAX_OUTPUT")
    num_ctx_min: int = Field(default=2048, validation_alias="KNOX_LLM_NUM_CTX_MIN")
    num_ctx_max: int = Field(default=131072, validation_alias="KNOX_LLM_NUM_CTX_MAX")
    context_window_cap: int | None = Field(default=None, validation_alias="KNOX_LLM_CONTEXT_WINDOW_CAP")
    role_models: dict[str, str] = Field(default_factory=dict)
    data_format: LlmDataFormat = Field(default=LlmDataFormat.JSON, validation_alias="KNOX_LLM_DATA_FORMAT")
    output_retries: int = Field(default=2, validation_alias="KNOX_LLM_OUTPUT_RETRIES")
    reasoning_effort: str | None = Field(default=None, validation_alias="KNOX_LLM_REASONING_EFFORT")


class BudgetConfig(_KnoxBaseModel):
    """Token budgets and structural limits for LLM operations.

    Sizing strategy:
    - Input chunks: ``clamp(context_window - reserve, chunk_size_min, chunk_size_max)`` (knox-7sp)
    - LLM output reserves: per-phase absolute ``*_output_max`` (knox-vxd).  Absolute
      reserves scale correctly to large context windows; an earlier ratio-based
      design (``*_output_ratio`` / ``*_output_min``) was specified but never
      implemented and has been removed.
    """

    concepts_per_batch: int = Field(default=8, validation_alias="KNOX_BUDGET_CONCEPTS_PER_BATCH")
    verify_min_ngram: int = 3
    wicer_max_iterations: int = Field(default=3, validation_alias="KNOX_BUDGET_WICER_MAX_ITERATIONS")
    min_content_length: int = Field(default=50, validation_alias="KNOX_BUDGET_MIN_CONTENT_LENGTH")
    min_concept_sources: int = Field(default=2, validation_alias="KNOX_BUDGET_MIN_CONCEPT_SOURCES")
    max_concept_pages: int = Field(default=50, validation_alias="KNOX_BUDGET_MAX_CONCEPT_PAGES")
    chunk_size_min: int = Field(default=2048, validation_alias="KNOX_BUDGET_CHUNK_SIZE_MIN")
    chunk_size_max: int = Field(default=131072, validation_alias="KNOX_BUDGET_CHUNK_SIZE_MAX")
    extract_output_max: int = Field(default=8192, validation_alias="KNOX_BUDGET_EXTRACT_OUTPUT_MAX")
    gen_output_max: int = Field(default=16384, validation_alias="KNOX_BUDGET_GEN_OUTPUT_MAX")
    overview_output_max: int = Field(default=4096, validation_alias="KNOX_BUDGET_OVERVIEW_OUTPUT_MAX")
    verify_output_max: int = 4096
    lint_output_max: int = 8192
    wicer_probe_output_max: int = 4096
    wicer_eval_output_max: int = 4096
    summary_max_content_chars: int = 8000
    summary_target_chars: int = 600
    summary_skip_single_child: bool = True
    summary_chunk_threshold: int = 10
    capture_source_char_max: int = Field(default=4000, validation_alias="KNOX_BUDGET_CAPTURE_SOURCE_CHAR_MAX")


class OutputConfig(_KnoxBaseModel):
    """Configuration for output formatting and rendering.

    ``format`` defaults to ``okf`` (Open Knowledge Format v0.1).
    """

    format: OutputFormat = Field(default=OutputFormat.OKF, validation_alias="KNOX_FORMAT")
    flavor: str | None = Field(default=None, validation_alias="KNOX_OUTPUT_FLAVOR")
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    rumdl_config: Path | None = Field(default=None, validation_alias="KNOX_RUMDL_CONFIG")
    max_related_links: int = Field(default=15, validation_alias="KNOX_OUTPUT_MAX_RELATED_LINKS")


class EmbeddingConfig(_KnoxBaseModel):
    """Configuration for embedding models.

    Dimensions are auto-detected from the model at runtime — ``dim`` is a
    fallback used by the dedup orchestrator when the provider hasn't been
    instantiated yet.
    """

    model: str | None = Field(default=None, validation_alias="KNOX_EMBEDDING_MODEL")
    batch_size: int = Field(default=10, validation_alias="KNOX_EMBEDDING_BATCH_SIZE")
    dim: int = 384


class PdfConfig(_KnoxBaseModel):
    """Configuration for PDF parsing to prevent OOM on large scanned PDFs.

    When a PDF exceeds *max_pages_before_chunking*, Docling processes it in
    overlapping page-range segments to keep ONNX/OCR memory bounded.
    """

    max_pages_before_chunking: int = Field(default=50, validation_alias="KNOX_PDF_MAX_PAGES_BEFORE_CHUNKING")
    page_chunk_size: int = Field(default=30, validation_alias="KNOX_PDF_PAGE_CHUNK_SIZE")
    page_chunk_overlap: int = Field(default=5, validation_alias="KNOX_PDF_PAGE_CHUNK_OVERLAP")


# ── Other section configs ────────────────────────────────────────────────


class PathsConfig(_KnoxBaseModel):
    """Vault path overrides (all optional — resolved from base_dir at runtime)."""

    wiki_dir: str | None = Field(default=None, validation_alias="KNOX_WIKI_DIR")
    sources_dir: str = Field(default="sources", validation_alias="KNOX_SOURCES_DIR")
    output_dir: str = Field(default="wiki", validation_alias="KNOX_OUTPUT_DIR")
    cache_dir: str = Field(default=CACHE_DIR, validation_alias="KNOX_CACHE_DIR")
    state_dir: str = Field(default=KNOX_DIR, validation_alias="KNOX_STATE_DIR")


class LintConfig(_KnoxBaseModel):
    """Lint configuration. ``enabled=None`` means all structural rules."""

    enabled: list[str] | None = None


class EvalConfig(_KnoxBaseModel):
    """Evaluation configuration."""

    threshold: float | None = Field(default=None, validation_alias="KNOX_EVAL_THRESHOLD")
    history_size: int = 20


class DedupConfig(_KnoxBaseModel):
    """Deduplication configuration (opt-in by default)."""

    enabled: bool = Field(default=False, validation_alias="KNOX_DEDUP_ENABLED")
    matchers: str = Field(default="embedding,minhash", validation_alias="KNOX_DEDUP_MATCHERS")
    threshold_identical: float = Field(default=0.95, validation_alias="KNOX_DEDUP_THRESHOLD_IDENTICAL")
    threshold_similar: float = Field(default=0.80, validation_alias="KNOX_DEDUP_THRESHOLD_SIMILAR")
    model: str = Field(default="", validation_alias="KNOX_DEDUP_MODEL")
    minhash_lsh_threshold: float = Field(default=0.8, validation_alias="KNOX_DEDUP_MINHASH_LSH_THRESHOLD")


class TrustConfig(_KnoxBaseModel):
    """Privacy and trust configuration."""

    privacy_filter: bool = Field(default=True, validation_alias="KNOX_PRIVACY_FILTER")
    user_tracking: bool = Field(default=True, validation_alias="KNOX_USER_TRACKING")


class QueryConfig(_KnoxBaseModel):
    """Query command configuration."""

    format: QueryFormat = QueryFormat.TEXT


class SearchExpansionConfig(_KnoxBaseModel):
    enabled: bool = Field(default=False, validation_alias="KNOX_SEARCH_EXPANSION_ENABLED")
    variants: int = Field(default=3, validation_alias="KNOX_SEARCH_EXPANSION_VARIANTS")


class SearchCragConfig(_KnoxBaseModel):
    enabled: bool = Field(default=False, validation_alias="KNOX_SEARCH_CRAG_ENABLED")
    threshold: float = Field(default=0.3, validation_alias="KNOX_SEARCH_CRAG_THRESHOLD")


class SearchRerankerConfig(_KnoxBaseModel):
    enabled: bool = Field(default=False, validation_alias="KNOX_SEARCH_RERANKER_ENABLED")
    model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", validation_alias="KNOX_SEARCH_RERANKER_MODEL")
    top_k: int = Field(default=20, validation_alias="KNOX_SEARCH_RERANKER_TOP_K")


class SearchHydeConfig(_KnoxBaseModel):
    enabled: bool = Field(default=False, validation_alias="KNOX_SEARCH_HYDE_ENABLED")


class SearchContextualConfig(_KnoxBaseModel):
    enabled: bool = Field(default=False, validation_alias="KNOX_SEARCH_CONTEXTUAL_ENABLED")
    model: str = Field(default="", validation_alias="KNOX_SEARCH_CONTEXTUAL_MODEL")


class SearchSynthesisTokensConfig(_KnoxBaseModel):
    brief: int = Field(default=300, validation_alias="KNOX_SEARCH_SYNTHESIS_TOKENS_BRIEF")
    standard: int = Field(default=800, validation_alias="KNOX_SEARCH_SYNTHESIS_TOKENS_STANDARD")
    comprehensive: int = Field(default=2000, validation_alias="KNOX_SEARCH_SYNTHESIS_TOKENS_COMPREHENSIVE")


class SearchSynthesisConfig(_KnoxBaseModel):
    verbosity: Verbosity = Field(default=Verbosity.STANDARD, validation_alias="KNOX_SEARCH_SYNTHESIS_VERBOSITY")
    tokens: SearchSynthesisTokensConfig = Field(default_factory=SearchSynthesisTokensConfig)


class SearchConfig(_KnoxBaseModel):
    expansion: SearchExpansionConfig = Field(default_factory=SearchExpansionConfig)
    crag: SearchCragConfig = Field(default_factory=SearchCragConfig)
    reranker: SearchRerankerConfig = Field(default_factory=SearchRerankerConfig)
    hyde: SearchHydeConfig = Field(default_factory=SearchHydeConfig)
    contextual: SearchContextualConfig = Field(default_factory=SearchContextualConfig)
    synthesis: SearchSynthesisConfig = Field(default_factory=SearchSynthesisConfig)


class OrganizeConfig(_KnoxBaseModel):
    """Configuration for ``knox organize`` LLM-based file categorization."""

    max_depth: int = Field(default=3, validation_alias="KNOX_ORGANIZE_MAX_DEPTH")
    min_subcategories: int = Field(default=2, validation_alias="KNOX_ORGANIZE_MIN_SUBCATEGORIES")
    min_files_per_category: int = 2
    collapse_single_file: bool = Field(default=True, validation_alias="KNOX_ORGANIZE_COLLAPSE_SINGLE_FILE")


class IgnoreConfig(_KnoxBaseModel):
    patterns: list[str] = Field(default_factory=list)


class DisplayConfig(_KnoxBaseModel):
    """Display/preview truncation limits.

    Each field caps how many items or how much text a single rendered
    section (index, log report, related-titles list, etc.) shows.
    Limits are deliberately configurable so users with long source lists
    or large captured notes can widen them without touching code.
    """

    source_names: int = Field(default=20, validation_alias="KNOX_DISPLAY_SOURCE_NAMES")
    related_titles: int = Field(default=15, validation_alias="KNOX_DISPLAY_RELATED_TITLES")
    doc_headings: int = Field(default=5, validation_alias="KNOX_DISPLAY_DOC_HEADINGS")
    concept_names: int = Field(default=3, validation_alias="KNOX_DISPLAY_CONCEPT_NAMES")
    memory_entries: int = Field(default=10, validation_alias="KNOX_DISPLAY_MEMORY_ENTRIES")
    excerpt_chars: int = Field(default=300, validation_alias="KNOX_DISPLAY_EXCERPT_CHARS")
    error_chars: int = Field(default=200, validation_alias="KNOX_DISPLAY_ERROR_CHARS")
    json_chars: int = Field(default=300, validation_alias="KNOX_DISPLAY_JSON_CHARS")
    cli_element_chars: int = Field(default=80, validation_alias="KNOX_DISPLAY_CLI_ELEMENT_CHARS")
    tier_sample_size: int = Field(default=5, validation_alias="KNOX_DISPLAY_TIER_SAMPLE_SIZE")
    cli_arg_preview: int = Field(default=4, validation_alias="KNOX_DISPLAY_CLI_ARG_PREVIEW")
    log_error_chars: int = Field(default=500, validation_alias="KNOX_DISPLAY_LOG_ERROR_CHARS")
    lint_feedback_chars: int = Field(default=200, validation_alias="KNOX_DISPLAY_LINT_FEEDBACK_CHARS")
    memory_query_chars: int = Field(default=50, validation_alias="KNOX_DISPLAY_MEMORY_QUERY_CHARS")
    mcp_element_chars: int = Field(default=500, validation_alias="KNOX_DISPLAY_MCP_ELEMENT_CHARS")
    wiki_topics: int = Field(default=30, validation_alias="KNOX_DISPLAY_WIKI_TOPICS")
    query_expansions: int = Field(default=7, validation_alias="KNOX_DISPLAY_QUERY_EXPANSIONS")
    web_nodes: int = Field(default=10, validation_alias="KNOX_DISPLAY_WEB_NODES")
    search_snippet_chars: int = Field(default=300, validation_alias="KNOX_DISPLAY_SEARCH_SNIPPET_CHARS")


class NotesConfig(_KnoxBaseModel):
    file: str = Field(default="NOTES.md", validation_alias="KNOX_NOTES_FILE")


class IngestConfig(_KnoxBaseModel):
    parser: str = Field(default="docling", validation_alias="KNOX_INGEST_PARSER")


class DownloaderConfig(_KnoxBaseModel):
    undetected: bool = Field(default=True, validation_alias="KNOX_DOWNLOADER_UNDETECTED")


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
    image_mode: ImageMode = Field(default=ImageMode.FILE, validation_alias="KNOX_IMAGE_MODE")
    vlm_min_image_bytes: int = 2048
    vlm_min_image_dimension: int = 64
    mode: str = "full"
    workers: int = Field(default=1, validation_alias="KNOX_WORKERS")
    no_cache: bool = Field(default=False, validation_alias="KNOX_NO_CACHE")
    force: bool = Field(default=False, validation_alias="KNOX_FORCE")
    in_place_sources: bool = Field(default=False, validation_alias="KNOX_IN_PLACE_SOURCES")
    memory_enabled: bool = Field(default=True, validation_alias="KNOX_MEMORY_ENABLED")
    memory_half_life_days: float = Field(default=90.0, validation_alias="KNOX_MEMORY_HALF_LIFE_DAYS")
    memory_decay_threshold: float = Field(default=0.3, validation_alias="KNOX_MEMORY_DECAY_THRESHOLD")
    review: bool = False
    bypass_review: bool = False
    provenance: bool = False
    private: bool = False
    verify: bool = True
    synthesize: bool = True
    index: bool = True
    lint: bool = Field(default=False, validation_alias="KNOX_LINT")

    # Nested sub-configs
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    pdf: PdfConfig = Field(default_factory=PdfConfig)
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
    organize: OrganizeConfig = Field(default_factory=OrganizeConfig)
    ignore: IgnoreConfig = Field(default_factory=IgnoreConfig)
    hooks: dict[str, Any] = Field(default_factory=dict)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    notes: NotesConfig = Field(default_factory=NotesConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    downloader: DownloaderConfig = Field(default_factory=DownloaderConfig)
