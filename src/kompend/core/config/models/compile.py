"""Compile section models (``[compile]`` and its nested sub-sections)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from kompend.core.config.enums import Compiler, ImageMode, Tier
from kompend.core.config.models._base import _KompendBaseModel
from kompend.core.config.models.llm import EmbeddingConfig, LLMConfig
from kompend.core.config.models.output import OutputConfig
from kompend.core.keys import MANDATORY_KINDS, EnvVar


class BudgetConfig(_KompendBaseModel):
    """Token budgets and structural limits for LLM operations.

    Sizing strategy:
    - Input chunks: ``clamp(context_window - reserve, chunk_size_min, chunk_size_max)``.
    - No output ceiling exists. ``max_tokens`` is never sent to a model: an
      answer is bounded by the context window minus its prompt, and the
      planners leave ``ContextWindow.answer_reserve`` free for it (see
      ``kompend.llm.context_window``).
    """

    #: Maximum refinement rounds of the WiCER loop. ``0`` means "until the
    #: loop converges" — the loop stops on its own when it finds no new
    #: facts, when all probes pass, or when the wall-clock budget
    #: (``wicer_max_seconds``) is spent. A positive value is an explicit
    #: user-set limit.
    wicer_max_iterations: int = Field(default=0, validation_alias=EnvVar.BUDGET_WICER_MAX_ITERATIONS)
    #: Wall-clock budget for the WiCER refinement loop, in seconds. ``0``
    #: disables the limit. A timeout is a legitimate stop reason; it never
    #: discards the wiki that was already compiled.
    wicer_max_seconds: int = Field(default=1800, ge=0, validation_alias=EnvVar.BUDGET_WICER_MAX_SECONDS)
    #: Warning threshold for the body length of a group/node. Content below
    #: it is kept — the value never excludes anything.
    min_content_length: int = Field(default=50, validation_alias=EnvVar.BUDGET_MIN_CONTENT_LENGTH)
    min_concept_sources: int = Field(default=2, validation_alias=EnvVar.BUDGET_MIN_CONCEPT_SOURCES)
    #: Minimum distinct sources an entity must span before the two-phase
    #: compiler emits a standalone entity page. Entities below it stay folded
    #: into their source summary pages — the mention is not lost.
    entity_min_sources: int = Field(default=2, ge=1)
    #: Per entity-type overrides of :attr:`entity_min_sources`, e.g.
    #: ``{"Organization": 1}``. Types are matched case-insensitively; types
    #: not listed use the default.
    entity_min_sources_by_type: dict[str, Annotated[int, Field(ge=1)]] = Field(default_factory=dict)
    #: Warning threshold for the number of concept pages. Above it every
    #: candidate is compiled — the value never drops a candidate.
    max_concept_pages: int = Field(default=50, validation_alias=EnvVar.BUDGET_MAX_CONCEPT_PAGES)
    chunk_size_min: int = Field(default=2048, validation_alias=EnvVar.BUDGET_CHUNK_SIZE_MIN)
    chunk_size_max: int = Field(default=131072, validation_alias=EnvVar.BUDGET_CHUNK_SIZE_MAX)
    #: Characters per payload window of the summarizer. One window
    #: costs one summarize call, so a bigger window means fewer calls (faster)
    #: — it never drops content.
    summary_window_chars: int = Field(default=2000, ge=1, validation_alias=EnvVar.BUDGET_SUMMARY_WINDOW_CHARS)
    summary_skip_single_child: bool = True
    summary_chunk_threshold: int = 10
    summary_requests_per_node: int = Field(default=2, ge=1, validation_alias=EnvVar.BUDGET_SUMMARY_REQUESTS_PER_NODE)
    #: LLM requests budgeted per tag group in one agentic distill session.
    #: Raise for models that spend turns on prose or notebook tools before
    #: they reach ``save_page()`` - the session's retry budgets are added on
    #: top of this factor, never folded into it.
    distill_requests_per_group: int = Field(
        default=3,
        ge=1,
        validation_alias=EnvVar.BUDGET_DISTILL_REQUESTS_PER_GROUP,
    )
    #: How many concepts one agentic generation segment owns. Each pass of
    #: the segment loop re-sends the segment content, so a segment that owns
    #: a whole document's concept list costs one prefill per batch of pages.
    concepts_per_segment: int = Field(default=12, validation_alias=EnvVar.BUDGET_CONCEPTS_PER_SEGMENT)
    #: How many times a generation segment that overflows the context window
    #: is halved and re-queued (Loop-1 deterministic shrink) before it fails
    #: loudly. ``0`` disables the shrink — an oversized segment fails like
    #: it used to. Each split strictly reduces the unit, so the loop
    #: terminates by construction.
    segment_shrink_attempts: int = Field(default=2, ge=0, validation_alias=EnvVar.BUDGET_SEGMENT_SHRINK_ATTEMPTS)
    summary_session_batch_size: int = Field(default=20, ge=1, validation_alias=EnvVar.BUDGET_SUMMARY_SESSION_BATCH_SIZE)
    summary_session_stall_retries: int = Field(default=1, ge=0, validation_alias=EnvVar.BUDGET_SUMMARY_SESSION_STALL_RETRIES)
    #: Traversal planning for the chunk-tree summarizer. ``"off"``
    #: summarizes the whole tree; ``"auto"`` plans one level at a time with
    #: the decision model, skipping branches the task does not need.
    summary_traversal: Literal["off", "auto"] = Field(default="off", validation_alias=EnvVar.BUDGET_SUMMARY_TRAVERSAL)
    #: Most children one planned level may expand. The plan is a contract:
    #: a window that spends past the remaining budget is sent back.
    summary_traversal_max_nodes: int = Field(default=200, ge=1, validation_alias=EnvVar.BUDGET_SUMMARY_TRAVERSAL_MAX_NODES)
    #: Most summarizer windows one planned tree may spend.
    summary_traversal_max_windows: int = Field(default=2000, ge=1, validation_alias=EnvVar.BUDGET_SUMMARY_TRAVERSAL_MAX_WINDOWS)
    #: Trees below this many nodes skip planning — the planner is pure
    #: overhead on a tree that summarizes in one pass anyway.
    summary_traversal_min_nodes: int = Field(default=50, ge=1, validation_alias=EnvVar.BUDGET_SUMMARY_TRAVERSAL_MIN_NODES)
    #: Decision model override for traversal planning; falls back to
    #: ``decision.model`` when unset.
    summary_traversal_model: str | None = Field(default=None, validation_alias=EnvVar.BUDGET_SUMMARY_TRAVERSAL_MODEL)
    #: Characters served per ``read_source()`` call in the write-page
    #: synthesis. A longer source is read in several windows (the agent
    #: pages through it) — it is never shortened.
    capture_source_char_max: int = Field(default=4000, validation_alias=EnvVar.BUDGET_CAPTURE_SOURCE_CHAR_MAX)
    concept_merge_threshold: float = Field(default=0.82, validation_alias=EnvVar.BUDGET_CONCEPT_MERGE_THRESHOLD)
    concept_string_merge_threshold: float = Field(default=0.70, validation_alias=EnvVar.BUDGET_CONCEPT_STRING_MERGE_THRESHOLD)
    #: Largest fraction of the concept pool one merge may absorb. A merge
    #: that would remove more than this is refused: same-domain text sits above
    #: the threshold for clearly distinct concepts, so a transitive chain can
    #: swallow most of the corpus in one go. ``1.0`` disables the guard.
    concept_merge_max_reduction: float = Field(default=0.5, gt=0.0, le=1.0, validation_alias=EnvVar.BUDGET_CONCEPT_MERGE_MAX_REDUCTION)
    #: Largest number of sources one merged concept may hold. A cluster that
    #: spans more sources than this is refused - the p863 corpus absorbed 98
    #: sources under one canonical name. ``0`` disables the guard.
    concept_merge_max_sources: int = Field(default=24, ge=0, validation_alias=EnvVar.BUDGET_CONCEPT_MERGE_MAX_SOURCES)
    jaccard_min_token_len: int = Field(default=1, validation_alias=EnvVar.BUDGET_JACCARD_MIN_TOKEN_LEN)
    edge_min_title_len: int = Field(default=4, validation_alias=EnvVar.BUDGET_EDGE_MIN_TITLE_LEN)


class ExtractionConfig(_KompendBaseModel):
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


class SkillConfig(_KompendBaseModel):
    """A named skill — user-defined instructions injected into the PydanticAI agent's system prompt.

    Skills are defined under ``[compile.skills.<name>]`` in config:

    .. code-block:: toml

        [compile.skills."domain-terms"]
        instructions = '''...'''  # free-form skill instructions
    """

    instructions: str


class CompileConfig(_KompendBaseModel):
    """Compile section: top-level flags and nested sub-configs."""

    # Hidden duplicates of paths.* (kept for backward compat, stripped from scaffold)
    sources_dir: str = "sources"
    output_dir: str = "wiki"

    compiler: Compiler = Compiler.GRAPH
    image_mode: ImageMode = Field(default=ImageMode.FILE, validation_alias=EnvVar.IMAGE_MODE)
    vlm_min_image_bytes: int = 2048
    vlm_min_image_dimension: int = 64
    vlm_context_chars: int = 1200
    # Model-free pre-filter ahead of the caption call. Off by default: the
    # byte/dimension floors above remain the only always-on filter, so the
    # rollout is opt-in and gate-off is byte-for-byte the old behaviour.
    vlm_heuristic_filter_enabled: bool = Field(default=False, validation_alias=EnvVar.VLM_HEURISTIC_FILTER_ENABLED)
    # A canvas this elongated, with a short edge this thin, is a line/border.
    vlm_heuristic_min_aspect_ratio: float = Field(default=20.0, validation_alias=EnvVar.VLM_HEURISTIC_MIN_ASPECT_RATIO)
    vlm_heuristic_max_thin_edge: int = Field(default=96, validation_alias=EnvVar.VLM_HEURISTIC_MAX_THIN_EDGE)
    # Bounded pixel sample: the long edge and the decoded pixel budget.
    vlm_heuristic_max_analysis_edge: int = Field(default=320, validation_alias=EnvVar.VLM_HEURISTIC_MAX_ANALYSIS_EDGE)
    vlm_heuristic_max_analysis_pixels: int = Field(default=2_000_000, validation_alias=EnvVar.VLM_HEURISTIC_MAX_ANALYSIS_PIXELS)
    # Grayscale standard deviation at or below this is a solid/near-solid fill.
    vlm_heuristic_uniform_stddev: float = Field(default=1.0, validation_alias=EnvVar.VLM_HEURISTIC_UNIFORM_STDDEV)
    # Near-blank: low variance and almost no edge pixels.
    vlm_heuristic_near_blank_stddev: float = Field(default=4.0, validation_alias=EnvVar.VLM_HEURISTIC_NEAR_BLANK_STDDEV)
    vlm_heuristic_near_blank_edge_density: float = Field(default=0.001, validation_alias=EnvVar.VLM_HEURISTIC_NEAR_BLANK_EDGE_DENSITY)
    # Transparency is only a hint; it never drops a figure on its own.
    vlm_heuristic_transparency_ratio: float = Field(default=0.995, validation_alias=EnvVar.VLM_HEURISTIC_TRANSPARENCY_RATIO)
    vlm_heuristic_min_confidence: float = Field(default=0.9, validation_alias=EnvVar.VLM_HEURISTIC_MIN_CONFIDENCE)
    workers: int = Field(default=1, validation_alias=EnvVar.WORKERS)
    no_cache: bool = Field(default=False, validation_alias=EnvVar.NO_CACHE)
    # Re-parse every source instead of reusing the cached parse: sources are
    # re-extracted and re-chunked, cached LLM results are kept.
    reparse: bool = Field(default=False, validation_alias=EnvVar.REPARSE)
    # Reuse cached extractions/summaries/pages across model switches: cache
    # keys drop the model component, so prior-model output is replayed as-is.
    # Pages keep the model tag they were generated with (metadata "model").
    cache_ignore_model: bool = Field(default=False, validation_alias=EnvVar.CACHE_IGNORE_MODEL)
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
    start_at_tier: Tier = Field(default=Tier.INDEX, validation_alias=EnvVar.START_AT_TIER)
    stop_at_tier: Tier = Field(default=Tier.OUTPUT, validation_alias=EnvVar.STOP_AT_TIER)
    respect_gitignore: bool = True
    source_tags: list[str] = Field(default_factory=list)
    tier_state_file: str | None = None

    # Nested sub-configs
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    skills: dict[str, SkillConfig] = Field(default_factory=dict)
