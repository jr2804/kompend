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
COMPILE_PAGE_KIND: Final[str] = "compile.page_kind"
COMPILE_START_AT_TIER: Final[str] = "compile.start_at_tier"
COMPILE_STOP_AT_TIER: Final[str] = "compile.stop_at_tier"
COMPILE_CHUNK_SIZE: Final[str] = "compile.chunk_size"
COMPILE_FORCE: Final[str] = "compile.force"
COMPILE_NO_CACHE: Final[str] = "compile.no_cache"
COMPILE_LINT: Final[str] = "compile.lint"
COMPILE_VERIFY: Final[str] = "compile.verify"
COMPILE_INDEX: Final[str] = "compile.index"
COMPILE_SYNTHESIZE: Final[str] = "compile.synthesize"
COMPILE_REVIEW: Final[str] = "compile.review"
COMPILE_PROVENANCE: Final[str] = "compile.provenance"
COMPILE_PRIVATE: Final[str] = "compile.private"
COMPILE_RESOLVE_RELATIONS: Final[str] = "compile.resolve_relations"
COMPILE_RELATION_CONFIDENCE_THRESHOLD: Final[str] = "compile.relation_confidence_threshold"
COMPILE_RESPECT_GITIGNORE: Final[str] = "compile.respect_gitignore"
COMPILE_MEMORY_ENABLED: Final[str] = "compile.memory_enabled"
COMPILE_MEMORY_HALF_LIFE_DAYS: Final[str] = "compile.memory_half_life_days"
COMPILE_MEMORY_DECAY_THRESHOLD: Final[str] = "compile.memory_decay_threshold"
COMPILE_TIER_STATE_PATH: Final[str] = "compile.tier_state_path"

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
COMPILE_BUDGET_EXTRACT_OUTPUT_RATIO: Final[str] = "compile.budget.extract_output_ratio"
COMPILE_BUDGET_EXTRACT_OUTPUT_MIN: Final[str] = "compile.budget.extract_output_min"
COMPILE_BUDGET_EXTRACT_OUTPUT_MAX: Final[str] = "compile.budget.extract_output_max"
COMPILE_BUDGET_GEN_OUTPUT_MAX: Final[str] = "compile.budget.gen_output_max"
COMPILE_BUDGET_OVERVIEW_OUTPUT_RATIO: Final[str] = "compile.budget.overview_output_ratio"
COMPILE_BUDGET_OVERVIEW_OUTPUT_MIN: Final[str] = "compile.budget.overview_output_min"
COMPILE_BUDGET_OVERVIEW_OUTPUT_MAX: Final[str] = "compile.budget.overview_output_max"
COMPILE_BUDGET_VERIFY_OUTPUT_RATIO: Final[str] = "compile.budget.verify_output_ratio"
COMPILE_BUDGET_VERIFY_OUTPUT_MIN: Final[str] = "compile.budget.verify_output_min"
COMPILE_BUDGET_VERIFY_OUTPUT_MAX: Final[str] = "compile.budget.verify_output_max"
COMPILE_BUDGET_LINT_OUTPUT_MAX: Final[str] = "compile.budget.lint_output_max"
COMPILE_BUDGET_WICER_PROBE_OUTPUT_MAX: Final[str] = "compile.budget.wicer_probe_output_max"
COMPILE_BUDGET_WICER_EVAL_OUTPUT_MAX: Final[str] = "compile.budget.wicer_eval_output_max"

# ── compile.output sub-section ──
COMPILE_OUTPUT_FLAVOR: Final[str] = "compile.output.flavor"
COMPILE_OUTPUT_FRONTMATTER: Final[str] = "compile.output.frontmatter"
COMPILE_OUTPUT_MAX_RELATED_LINKS: Final[str] = "compile.output.max_related_links"
COMPILE_OUTPUT_RUMDL_CONFIG: Final[str] = "compile.output.rumdl_config"

# ── trust/privacy ──
TRUST_PRIVACY_FILTER: Final[str] = "trust.privacy_filter"
