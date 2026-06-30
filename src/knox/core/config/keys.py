"""Single source of truth for all config dotted-key strings.

Every ``cfg.get("...")`` call must reference a constant from this module.
This prevents typos and makes config schema changes searchable in one place.

No ``"compile.xxx"`` or ``"paths.xxx"`` literal may appear outside this file.
"""

from typing import Final

# ── paths section ──
PATHS_WIKI_DIR: Final[str] = "paths.wiki_dir"
PATHS_SOURCES_DIR: Final[str] = "paths.sources_dir"
PATHS_OUTPUT_DIR: Final[str] = "paths.output_dir"
PATHS_CACHE_DIR: Final[str] = "paths.cache_dir"
PATHS_STATE_DIR: Final[str] = "paths.state_dir"

# ── compile section (flat keys) ──
COMPILE_SOURCES_DIR: Final[str] = "compile.sources_dir"
COMPILE_OUTPUT_DIR: Final[str] = "compile.output_dir"
COMPILE_CACHE_DIR: Final[str] = "compile.cache_dir"
COMPILE_FORMAT: Final[str] = "compile.format"
COMPILE_COMPILER: Final[str] = "compile.compiler"
COMPILE_WORKERS: Final[str] = "compile.workers"
COMPILE_IMAGE_MODE: Final[str] = "compile.image_mode"
COMPILE_VLM_MIN_IMAGE_BYTES: Final[str] = "compile.vlm_min_image_bytes"
COMPILE_VLM_MIN_IMAGE_DIMENSION: Final[str] = "compile.vlm_min_image_dimension"
COMPILE_PAGE_KIND: Final[str] = "compile.page_kind"
COMPILE_START_AT_TIER: Final[str] = "compile.start_at_tier"
COMPILE_STOP_AT_TIER: Final[str] = "compile.stop_at_tier"
COMPILE_CHUNK_SIZE: Final[str] = "compile.chunk_size"
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
COMPILE_MEMORY_HALF_LIFE_DAYS: Final[str] = "compile.memory_half_life_days"
COMPILE_MEMORY_DECAY_THRESHOLD: Final[str] = "compile.memory_decay_threshold"
COMPILE_TIER_STATE_PATH: Final[str] = "compile.tier_state_file"
# ── compile.llm sub-section ──
COMPILE_LLM_DEFAULT_MODEL: Final[str] = "compile.llm.default_model"
COMPILE_LLM_VLM_MODEL: Final[str] = "compile.llm.vlm_model"
COMPILE_LLM_FALLBACK_CHAIN: Final[str] = "compile.llm.fallback_chain"
COMPILE_LLM_ROLE_MODELS: Final[str] = "compile.llm.role_models"
COMPILE_LLM_FALLBACK_MAX_RETRIES: Final[str] = "compile.llm.fallback_max_retries"
COMPILE_LLM_TIMEOUT: Final[str] = "compile.llm.timeout"
COMPILE_LLM_FALLBACK_MAX_INPUT: Final[str] = "compile.llm.fallback_max_input"
COMPILE_LLM_FALLBACK_MAX_OUTPUT: Final[str] = "compile.llm.fallback_max_output"
COMPILE_LLM_NUM_CTX_MIN: Final[str] = "compile.llm.num_ctx_min"
COMPILE_LLM_NUM_CTX_MAX: Final[str] = "compile.llm.num_ctx_max"
COMPILE_LLM_CONTEXT_WINDOW_CAP: Final[str] = "compile.llm.context_window_cap"
COMPILE_LLM_API_BASE: Final[str] = "compile.llm.api_base"
COMPILE_LLM_API_KEY: Final[str] = "compile.llm.api_key"
COMPILE_LLM_DATA_FORMAT: Final[str] = "compile.llm.data_format"

# ── compile.pdf sub-section ──
COMPILE_PDF_MAX_PAGES_BEFORE_CHUNKING: Final[str] = "compile.pdf.max_pages_before_chunking"
COMPILE_PDF_PAGE_CHUNK_SIZE: Final[str] = "compile.pdf.page_chunk_size"
COMPILE_PDF_PAGE_CHUNK_OVERLAP: Final[str] = "compile.pdf.page_chunk_overlap"

# ── organize section ──
ORGANIZE_MAX_DEPTH: Final[str] = "organize.max_depth"
ORGANIZE_MIN_SUBCATEGORIES: Final[str] = "organize.min_subcategories"
ORGANIZE_COLLAPSE_SINGLE_FILE: Final[str] = "organize.collapse_single_file"

# ── compile.embedding sub-section ──
COMPILE_EMBEDDING_MODEL: Final[str] = "compile.embedding.model"
COMPILE_EMBEDDING_PROVIDER: Final[str] = "compile.embedding.provider"
COMPILE_EMBEDDING_BATCH_SIZE: Final[str] = "compile.embedding.batch_size"
COMPILE_EMBEDDING_DIM: Final[str] = "compile.embedding.dim"

# ── compile.budget sub-section ──
COMPILE_BUDGET_CONCEPTS_PER_BATCH: Final[str] = "compile.budget.concepts_per_batch"
COMPILE_BUDGET_VERIFY_MIN_NGRAM: Final[str] = "compile.budget.verify_min_ngram"
COMPILE_BUDGET_WICER_MAX_ITERATIONS: Final[str] = "compile.budget.wicer_max_iterations"
COMPILE_BUDGET_MIN_CONTENT_LENGTH: Final[str] = "compile.budget.min_content_length"
COMPILE_BUDGET_MIN_CONCEPT_SOURCES: Final[str] = "compile.budget.min_concept_sources"
COMPILE_BUDGET_MAX_CONCEPT_PAGES: Final[str] = "compile.budget.max_concept_pages"
COMPILE_BUDGET_CHUNK_SIZE_MIN: Final[str] = "compile.budget.chunk_size_min"
COMPILE_BUDGET_CHUNK_SIZE_MAX: Final[str] = "compile.budget.chunk_size_max"
COMPILE_BUDGET_EXTRACT_OUTPUT_MAX: Final[str] = "compile.budget.extract_output_max"
COMPILE_BUDGET_GEN_OUTPUT_MAX: Final[str] = "compile.budget.gen_output_max"
COMPILE_BUDGET_OVERVIEW_OUTPUT_MAX: Final[str] = "compile.budget.overview_output_max"
COMPILE_BUDGET_VERIFY_OUTPUT_MAX: Final[str] = "compile.budget.verify_output_max"
COMPILE_BUDGET_LINT_OUTPUT_MAX: Final[str] = "compile.budget.lint_output_max"
COMPILE_BUDGET_WICER_PROBE_OUTPUT_MAX: Final[str] = "compile.budget.wicer_probe_output_max"
COMPILE_BUDGET_WICER_EVAL_OUTPUT_MAX: Final[str] = "compile.budget.wicer_eval_output_max"

# Node summarization (Phase 2 — hierarchical chunk tree summaries)
COMPILE_BUDGET_SUMMARY_MAX_CONTENT_CHARS: Final[str] = "compile.budget.summary_max_content_chars"
COMPILE_BUDGET_SUMMARY_TARGET_CHARS: Final[str] = "compile.budget.summary_target_chars"
COMPILE_BUDGET_SUMMARY_SKIP_SINGLE_CHILD: Final[str] = "compile.budget.summary_skip_single_child"
COMPILE_BUDGET_SUMMARY_CHUNK_THRESHOLD: Final[str] = "compile.budget.summary_chunk_threshold"

# ── compile.output sub-section ──
COMPILE_OUTPUT_FLAVOR: Final[str] = "compile.output.flavor"
COMPILE_OUTPUT_FRONTMATTER: Final[str] = "compile.output.frontmatter"
COMPILE_OUTPUT_MAX_RELATED_LINKS: Final[str] = "compile.output.max_related_links"
COMPILE_OUTPUT_RUMDL_CONFIG: Final[str] = "compile.output.rumdl_config"

# ── dedup section ──
DEDUP_ENABLED: Final[str] = "dedup.enabled"
DEDUP_MATCHERS: Final[str] = "dedup.matchers"
DEDUP_THRESHOLD_IDENTICAL: Final[str] = "dedup.threshold_identical"
DEDUP_THRESHOLD_SIMILAR: Final[str] = "dedup.threshold_similar"
DEDUP_MODEL: Final[str] = "dedup.model"
DEDUP_MINHASH_LSH_THRESHOLD: Final[str] = "dedup.minhash_lsh_threshold"

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

# ── query section ──
QUERY_FORMAT: Final[str] = "query.format"
