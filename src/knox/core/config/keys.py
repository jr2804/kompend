"""Dotted-key constants for config lookups.

After the typed-config migration, most code paths use ``cfg.knox.<path>``
attribute access on :class:`ResolvedConfig` instead of importing these
constants. The remaining usages are:

- CLI override dicts (``overrides[KEY] = value``) — key identifier needed
- DEFAULTS dict lookups (``DEFAULTS.get(KEY)``) — key identifier needed
- Tests that build resolved configs and want the dotted string explicitly

The typed config (:mod:`knox.core.config._configs`) is the canonical schema
and should be preferred for new code.
"""

from typing import Final

from knox.core.env import EnvVar  # noqa: F401  # re-exported: importers use knox.core.config.keys.EnvVar

# ── paths section ──
PATHS_WIKI_DIR: Final[str] = "paths.wiki_dir"
PATHS_SOURCES_DIR: Final[str] = "paths.sources_dir"
PATHS_OUTPUT_DIR: Final[str] = "paths.output_dir"
PATHS_STATE_DIR: Final[str] = "paths.state_dir"

# ── ingest section ──
INGEST_PARSER: Final[str] = "ingest.parser"

# ── downloader section ──
DOWNLOADER_UNDETECTED: Final[str] = "downloader.undetected"

# ── export section ──
EXPORT_ANNOTATIONS: Final[str] = "export.annotations"
EXPORT_SUMMARY: Final[str] = "export.summary"

# ── compile section (flat keys) ──
COMPILE_SOURCES_DIR: Final[str] = "compile.sources_dir"
COMPILE_OUTPUT_DIR: Final[str] = "compile.output_dir"
COMPILE_CACHE_DIR: Final[str] = "compile.cache_dir"
COMPILE_COMPILER: Final[str] = "compile.compiler"
COMPILE_WORKERS: Final[str] = "compile.workers"
COMPILE_START_AT_TIER: Final[str] = "compile.start_at_tier"
COMPILE_STOP_AT_TIER: Final[str] = "compile.stop_at_tier"
COMPILE_FORCE: Final[str] = "compile.force"
COMPILE_IN_PLACE_SOURCES: Final[str] = "compile.in_place_sources"
COMPILE_NO_CACHE: Final[str] = "compile.no_cache"
COMPILE_LINT: Final[str] = "compile.lint"
COMPILE_VERIFY: Final[str] = "compile.verify"
COMPILE_INDEX: Final[str] = "compile.index"
COMPILE_SYNTHESIZE: Final[str] = "compile.synthesize"
COMPILE_REVIEW: Final[str] = "compile.review"
COMPILE_PROVENANCE: Final[str] = "compile.provenance"
COMPILE_PRIVATE: Final[str] = "compile.private"
COMPILE_RESPECT_GITIGNORE: Final[str] = "compile.respect_gitignore"
COMPILE_MEMORY_ENABLED: Final[str] = "compile.memory_enabled"
COMPILE_TIER_STATE_FILE: Final[str] = "compile.tier_state_file"

# ── compile.llm sub-section ──
COMPILE_LLM_DEFAULT_MODEL: Final[str] = "compile.llm.default_model"
COMPILE_LLM_VLM_MODEL: Final[str] = "compile.llm.vlm_model"
COMPILE_LLM_FALLBACK_CHAIN: Final[str] = "compile.llm.fallback_chain"
COMPILE_LLM_ROLE_MODELS: Final[str] = "compile.llm.role_models"
COMPILE_LLM_FALLBACK_MAX_RETRIES: Final[str] = "compile.llm.fallback_max_retries"
COMPILE_LLM_CONTEXT_WINDOW_CAP: Final[str] = "compile.llm.context_window_cap"
COMPILE_LLM_OUTPUT_RETRIES: Final[str] = "compile.llm.output_retries"

# ── organize section ──
ORGANIZE_MAX_DEPTH: Final[str] = "organize.max_depth"
ORGANIZE_MIN_SUBCATEGORIES: Final[str] = "organize.min_subcategories"
ORGANIZE_MIN_FILES_PER_CATEGORY: Final[str] = "organize.min_files_per_category"
ORGANIZE_COLLAPSE_SINGLE_FILE: Final[str] = "organize.collapse_single_file"

# ── compile.embedding sub-section ──
COMPILE_EMBEDDING_MODEL: Final[str] = "compile.embedding.model"

#: Mandatory page kinds — always allowed, cannot be deactivated or shadowed
#: by compile.extraction.categories (Karpathy taxonomy: concepts, entities,
#: summaries, comparisons).
MANDATORY_KINDS: frozenset[str] = frozenset({"concept", "entity", "summary", "comparison"})

# ── dedup section ──
DEDUP_ENABLED: Final[str] = "dedup.enabled"
DEDUP_MATCHERS: Final[str] = "dedup.matchers"
DEDUP_THRESHOLD_IDENTICAL: Final[str] = "dedup.threshold_identical"
DEDUP_THRESHOLD_SIMILAR: Final[str] = "dedup.threshold_similar"
DEDUP_MODEL: Final[str] = "dedup.model"
DEDUP_MINHASH_LSH_THRESHOLD: Final[str] = "dedup.minhash_lsh_threshold"

# ── lint section ──
LINT_FUZZY_THRESHOLD: Final[str] = "lint.fuzzy_threshold"

# ── trust/privacy ──
TRUST_PRIVACY_FILTER: Final[str] = "trust.privacy_filter"
TRUST_USER_TRACKING: Final[str] = "trust.user_tracking"

# ── search section ──
SEARCH_EXPANSION_ENABLED: Final[str] = "search.expansion.enabled"
SEARCH_EXPANSION_VARIANTS: Final[str] = "search.expansion.variants"
SEARCH_CRAG_ENABLED: Final[str] = "search.crag.enabled"
SEARCH_CRAG_THRESHOLD: Final[str] = "search.crag.threshold"
SEARCH_RERANKER_ENABLED: Final[str] = "search.reranker.enabled"
SEARCH_RERANKER_MODEL: Final[str] = "search.reranker.model"
SEARCH_RERANKER_TOP_K: Final[str] = "search.reranker.top_k"
SEARCH_HYDE_ENABLED: Final[str] = "search.hyde.enabled"
SEARCH_CONTEXTUAL_ENABLED: Final[str] = "search.contextual.enabled"
SEARCH_CONTEXTUAL_MODEL: Final[str] = "search.contextual.model"
SEARCH_SYNTHESIS_VERBOSITY: Final[str] = "search.synthesis.verbosity"
SEARCH_SYNTHESIS_TOKENS_BRIEF: Final[str] = "search.synthesis.tokens.brief"
SEARCH_SYNTHESIS_TOKENS_STANDARD: Final[str] = "search.synthesis.tokens.standard"
SEARCH_SYNTHESIS_TOKENS_COMPREHENSIVE: Final[str] = "search.synthesis.tokens.comprehensive"

# ── display section ──
DISPLAY_SOURCE_NAMES: Final[str] = "display.source_names"
DISPLAY_RELATED_TITLES: Final[str] = "display.related_titles"
DISPLAY_DOC_HEADINGS: Final[str] = "display.doc_headings"
DISPLAY_CONCEPT_NAMES: Final[str] = "display.concept_names"
DISPLAY_MEMORY_ENTRIES: Final[str] = "display.memory_entries"
DISPLAY_EXCERPT_CHARS: Final[str] = "display.excerpt_chars"
DISPLAY_ERROR_CHARS: Final[str] = "display.error_chars"
DISPLAY_JSON_CHARS: Final[str] = "display.json_chars"
DISPLAY_CLI_ELEMENT_CHARS: Final[str] = "display.cli_element_chars"
DISPLAY_TIER_SAMPLE_SIZE: Final[str] = "display.tier_sample_size"
DISPLAY_CLI_ARG_PREVIEW: Final[str] = "display.cli_arg_preview"
DISPLAY_LOG_ERROR_CHARS: Final[str] = "display.log_error_chars"
DISPLAY_LINT_FEEDBACK_CHARS: Final[str] = "display.lint_feedback_chars"
DISPLAY_MEMORY_QUERY_CHARS: Final[str] = "display.memory_query_chars"
DISPLAY_MCP_ELEMENT_CHARS: Final[str] = "display.mcp_element_chars"
DISPLAY_WIKI_TOPICS: Final[str] = "display.wiki_topics"
DISPLAY_QUERY_EXPANSIONS: Final[str] = "display.query_expansions"
DISPLAY_WEB_NODES: Final[str] = "display.web_nodes"
DISPLAY_SEARCH_SNIPPET_CHARS: Final[str] = "display.search_snippet_chars"

# ── agents section ──
AGENTS_COMPILE: Final[str] = "agents.compile"
AGENTS_QUERY: Final[str] = "agents.query"
AGENTS_CAPTURE: Final[str] = "agents.capture"
AGENTS_ORGANIZE: Final[str] = "agents.organize"
AGENTS_GENERIC: Final[str] = "agents.generic"
