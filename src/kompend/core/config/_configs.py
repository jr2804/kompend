"""Pydantic config models for kompend.

Single source of truth for the config schema: each field carries its type,
default, and optional env-var alias in one declaration. ``DEFAULTS`` and
``_ENV_MAP`` in ``_core.py`` are auto-generated from these models.

The six sub-configs used by ``PipelineConfig`` (``LLMConfig``, ``BudgetConfig``,
``OutputConfig``, ``EmbeddingConfig``, ``OrganizeConfig``) double
as runtime types — they are constructed in ``from_resolved()`` and held as
fields on ``PipelineConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Final

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from kompend.core.keys import MANDATORY_KINDS, EnvVar
from kompend.core.paths import CACHE_DIR, KOMP_DIR
from kompend.core.vocab import ExportMode, LintOutput, LlmDataFormat, Verbosity

#: Spellings that disable the context manager — the accepted "off" surface of
#: ``compile.llm.context_manager`` (env var and TOML arrive as strings).
_CONTEXT_OFF_WORDS: Final[frozenset[str]] = frozenset({"", "off", "false", "none", "no"})


class ContextManagerPreset(StrEnum):
    """Context-compaction strategy offered to every agent.

    - ``default``: summarizing compaction at 90% of the context window.
    - ``aggressive``: summarizing compaction at 70%, keeping 5 messages.
    - ``sliding``: sliding-window compaction (drop oldest, keep 50).
    """

    DEFAULT = "default"
    AGGRESSIVE = "aggressive"
    SLIDING = "sliding"


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


class LlmApiType(StrEnum):
    """Wire protocol the LLM endpoint speaks.

    - ``openai-chat``: ``/v1/chat/completions`` (legacy chat completions).
    - ``openai-responses``: ``/v1/responses`` (OpenAI's structured-responses API).
    - ``anthropic``: ``/v1/messages`` (Anthropic Messages API).
    """

    OPENAI_CHAT = "openai-chat"
    OPENAI_RESPONSES = "openai-responses"
    ANTHROPIC = "anthropic"


class LlmMemoryStore(StrEnum):
    """Where the agent notebook (``compile.llm.agent_memory``) lives.

    ``SESSION`` keeps the notebook in-process: it is fresh per stage run and
    dies with the process. ``VAULT`` backs the notebook with a persistent
    :class:`~kompend.agents.vault_memory.DirectoryMemoryStore` under the
    vault's :attr:`~kompend.core.paths.VaultPaths.agent_memory_dir` — a
    ``MEMORY.md`` index per agent plus topic files the model reads and
    writes through its memory tools.
    """

    SESSION = "session"
    VAULT = "vault"


class MatcherName(StrEnum):
    """Dedup matcher selection — the ``dedup.matchers`` vocabulary."""

    EMBEDDING = "embedding"
    MINHASH = "minhash"


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


class ParsecraftEffort(StrEnum):
    """Conversion effort for the MinerU sub-backend.

    Maps to parsecraft's ``effort`` option. ``flash`` is the verified default
    (text PDFs, weight-free CPU work); ``high`` and ``xhigh`` engage the VLM path
    and load MinerU's checkpoint.
    """

    FLASH = "flash"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class ParsecraftJudge(StrEnum):
    """Judge-tier routing for the parsecraft backend.

    - ``none`` (default): the deterministic rule table; fast, no model call.
    - ``ollama``: local ollama daemon, needs a decision-capable model.
    - ``zen`` / ``typesafe-ai``: cloud providers, credentialed via env.
    """

    NONE = "none"
    OLLAMA = "ollama"
    ZEN = "zen"
    TYPESAFE_AI = "typesafe-ai"


class ParsecraftPreference(StrEnum):
    """Tiebreak axis when several candidates tie on the constraint check.

    Maps directly to parsecraft's ``RoutingPreference``: ``speed`` favours the
    cheapest eligible backend, ``quality`` the strongest, ``balanced`` the
    default tiebreak.
    """

    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"


class ReasoningEffort(StrEnum):
    """Reasoning effort for providers with an explicit effort control.

    Members other than :attr:`DEFAULT` map onto pydantic-ai's ``ThinkingLevel``
    (``none`` → ``thinking=False``, the rest pass through verbatim).
    :attr:`DEFAULT` means "leave ``thinking`` unset" — the provider's own
    default applies, no setting is injected.
    """

    DEFAULT = "default"
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


def _coerce_context_manager(value: object) -> object:
    """Normalize a raw ``compile.llm.context_manager`` value for the schema.

    Strings are matched against the off-words (→ ``False``) and the preset
    members; anything else passes through for pydantic to validate.
    """
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _CONTEXT_OFF_WORDS:
            return False
        return ContextManagerPreset(normalized)
    return value


#: The default matcher set — single source for the model default, the
#: orchestrator's fallback and the dedup store's stamp.
DEFAULT_MATCHERS: Final[tuple[MatcherName, ...]] = (MatcherName.EMBEDDING, MatcherName.MINHASH)


class Flavor(StrEnum):
    """Markdown dialect rendered by the output tier.

    Canonical flavors; ``github`` is accepted as an alias for ``gfm`` (see
    :data:`FLAVOR_ALIASES`). ``pandoc``/``kramdown``/``azure_devops`` render
    the closest flavor rumdl supports for those targets.
    """

    OBSIDIAN = "obsidian"
    GFM = "gfm"
    MKDOCS = "mkdocs"
    MDX = "mdx"
    QUARTO = "quarto"
    MYST = "myst"
    STANDARD = "standard"
    PANDOC = "pandoc"
    KRAMDOWN = "kramdown"
    AZURE_DEVOPS = "azure_devops"


def parse_matchers(value: object) -> list[MatcherName]:
    """Parse a ``dedup.matchers`` value into matcher members.

    Accepts the comma-string form (env var, legacy TOML) and the array form;
    an unknown name raises instead of being skipped, and a non-string,
    non-sequence value raises ``TypeError`` (raw config dicts are a trust
    boundary — anything can be in them).
    """
    if isinstance(value, str):
        parts: list[str | MatcherName] = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        msg = f"dedup.matchers must be a comma-string or a list, got {type(value).__name__}"
        raise TypeError(msg)
    matchers: list[MatcherName] = []
    for part in parts:
        if not part:
            continue
        if isinstance(part, MatcherName):
            matchers.append(part)
            continue
        try:
            matchers.append(MatcherName(part.strip()))
        except ValueError as e:
            expected = ", ".join(member.value for member in MatcherName)
            msg = f"unknown dedup matcher {part!r}; expected one of: {expected}"
            raise ValueError(msg) from e
    return matchers


#: Accepted non-canonical spellings → canonical member.
FLAVOR_ALIASES: Final[dict[str, Flavor]] = {"github": Flavor.GFM}  # nocheck: str-registry — product-name aliases are an open input domain


# ── Compile section (top-level flags + nested sub-configs) ──────────────


class Compiler(StrEnum):
    """Compiler backend selection.

    - ``GRAPH``: GraphCompiler (heading-based grouping, default)
    - ``WICER``: WiCERCompiler (iterative compile-evaluate-refine loop)
    """

    GRAPH = "graph"
    WICER = "wicer"


class Tier(StrEnum):
    """Identifiers for the kompend 5-tier compilation pipeline.

    Tiers are cumulative: running a higher tier first ensures the
    state of every lower tier is up to date.
    """

    INDEX = "index"
    CHUNK = "chunk"
    RESOLVE = "resolve"
    COMPILE = "compile"
    OUTPUT = "output"

    @classmethod
    @cache
    def all_tiers(cls) -> tuple[TierInfo, ...]:
        """Return metadata for every tier in execution order."""
        return (
            TierInfo(
                Tier.INDEX,
                0,
                "Tier 0: index sources — record local files and download online URLs. Inputs: source_dir, online_sources. Outputs: tier_state entries.",
            ),
            TierInfo(
                Tier.CHUNK,
                1,
                "Tier 1: parse documents and split them into chunks. Inputs: indexed source files. Outputs: list of chunks (cached on disk).",
            ),
            TierInfo(
                Tier.RESOLVE,
                2,
                "Tier 2: resolve chunks — extract entities/links (LLM-dependent). Inputs: chunks. Outputs: resolved chunks with entities/relations.",
            ),
            TierInfo(
                Tier.COMPILE,
                3,
                "Tier 3: compile resolved chunks into a knowledge graph (CompiledWiki). Inputs: resolved chunks. Outputs: CompiledWiki snapshot.",
            ),
            TierInfo(
                Tier.OUTPUT,
                4,
                "Tier 4: render the compiled wiki to the output directory. Inputs: CompiledWiki snapshot. Outputs: rendered files (markdown/json/typst).",
            ),
        )

    @classmethod
    def level_of(cls, tier: Tier) -> int:
        """Return the numeric level of a tier."""
        return cls._info_by_tier()[tier].level

    @classmethod
    def description_of(cls, tier: Tier) -> str:
        """Return the human-readable description of a tier."""
        return cls._info_by_tier()[tier].description

    @classmethod
    @cache
    def _info_by_tier(cls) -> dict[Tier, TierInfo]:
        """Return a mapping from tier to its metadata for O(1) lookup."""
        return {ti.tier: ti for ti in cls.all_tiers()}


@dataclass(frozen=True)
class TierInfo:
    """Metadata for a single pipeline tier."""

    tier: Tier
    level: int
    description: str


class _KompendBaseModel(BaseModel):
    """Base for all config models.

    ``populate_by_name=True`` allows construction via field name (e.g.
    ``LLMConfig(default_model="x")``) even when a ``validation_alias`` is set
    (e.g. ``"KOMP_LLM_DEFAULT_MODEL"``).
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


# Env-var mapping: every field below carries its ``KOMP_*`` env override as a
# ``validation_alias``; ``_build_env_map`` in ``engine/_schema.py`` turns those
# aliases into the config-map automatically.

# ── Compile sub-configs ──────────────────────────────────────────────────


class RlmConfig(_KompendBaseModel):
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


class LLMConfig(_KompendBaseModel):
    """Configuration for LLM models, timeouts, and fallback behavior."""

    default_model: str | None = Field(default=None, validation_alias=EnvVar.LLM_DEFAULT_MODEL)
    vlm_model: str | None = Field(default=None, validation_alias=EnvVar.VLM_MODEL)
    fallback_chain: list[str] = Field(default_factory=list, validation_alias=EnvVar.LLM_FALLBACK_CHAIN)
    timeout: float = Field(default=120.0, validation_alias=EnvVar.LLM_TIMEOUT)
    fallback_max_input: int = Field(default=32768, validation_alias=EnvVar.LLM_FALLBACK_MAX_INPUT)
    fallback_max_output: int = Field(default=4096, validation_alias=EnvVar.LLM_FALLBACK_MAX_OUTPUT)
    context_window: int | None = Field(
        default=None, validation_alias=EnvVar.LLM_CONTEXT_WINDOW, description="Context window length override/cap applied to every discovered model"
    )
    data_format: LlmDataFormat = Field(default=LlmDataFormat.JSON, validation_alias=EnvVar.LLM_DATA_FORMAT)
    #: Whole-request retries after a *transient* failure (a timeout, a dropped
    #: connection, a 429/5xx). Distinct from ``tool_retries`` (a tool raising
    #: ``ModelRetry``) and from ``output_retries`` (a schema violation): those
    #: fix the request, a transient retry only repeats it. It lives on the agent
    #: base, so every stage - generation, lint, extraction, verification - obeys
    #: one policy instead of each loop inventing backoff.
    transient_retries: int = Field(default=2, validation_alias=EnvVar.LLM_TRANSIENT_RETRIES)
    #: First backoff in seconds for a transient retry; doubled per attempt.
    transient_backoff: float = Field(default=5.0, validation_alias=EnvVar.LLM_TRANSIENT_BACKOFF)
    #: Mark the conversation prefix for provider-side prompt caching. Only
    #: Anthropic-protocol models act on it (``anthropic_cache_messages``,
    #: the proxy-safe variant); a multi-pass agentic loop re-sends the same
    #: prefix, so this turns repeated prefill into cache reads. Disable it
    #: for a gateway that rejects the ``cache_control`` field.
    prompt_cache: bool = Field(default=True, validation_alias=EnvVar.LLM_PROMPT_CACHE)
    #: Output-validation retry budget for plain (single-shot) agents: how
    #: often a model answer may fail schema validation before the run fails.
    output_retries: int = Field(default=2, validation_alias=EnvVar.LLM_OUTPUT_RETRIES)
    output_retries_enabled: bool = Field(default=True, validation_alias=EnvVar.LLM_OUTPUT_RETRIES_ENABLED)
    #: Per-turn output-validation retry budget for tool-based (agentic) agents.
    #: Defaults higher than ``output_retries`` because agentic agents run
    #: multi-turn loops (fetch concepts → read segment → generate) where each
    #: turn can be invalidated; 2 retries is too tight for the typical
    #: small-model interaction.  Does NOT govern tool-call retries — those
    #: have their own budget (``tool_retries``).
    agentic_output_retries: int = Field(default=4, validation_alias=EnvVar.LLM_AGENTIC_OUTPUT_RETRIES)
    #: Tool-call retry budget for every agent (a tool that keeps raising
    #: ``ModelRetry`` — a missing file, an invalid argument).  Separate from
    #: the output-validation budgets so a flaky tool can be given more trials
    #: without also granting a model more schema-validation attempts.
    #: Defaults to 3: one hallucinated tool call must not abort a run — the
    #: ``ModelRetry`` feedback is usually enough for the model to correct
    #: itself.
    tool_retries: int = Field(default=3, validation_alias=EnvVar.LLM_TOOL_RETRIES)
    #: Offer the agentic memory capability (a per-agent notebook the model
    #: can write to and read from) to the stages where ONE agent instance
    #: handles MANY outputs — summarisation, verification, remediation and
    #: distillation. Stages that build a fresh agent per unit (generation,
    #: capture, query, indexing) gain nothing from a notebook: the memory
    #: dies with the agent.  ON by default: a stage that sees many outputs
    #: in one session is the only place a notebook pays off.
    agent_memory: bool = Field(default=True, validation_alias=EnvVar.LLM_AGENT_MEMORY)
    #: Backend for the agent notebook. ``vault`` (default) persists it under
    #: the vault's agent-memory directory: a ``MEMORY.md`` index per agent
    #: (injected into prompts, capped) plus topic files behind the memory
    #: tools. The agent writes that notebook itself, so it is not part of the
    #: summary cache key - editing it leaves cached summaries as they are.
    #: ``session`` keeps it in process - fresh per stage run.  The vault
    #: backend backs every notebook stage - summarisation, verification,
    #: remediation and distillation.
    memory_store: LlmMemoryStore = Field(default=LlmMemoryStore.VAULT, validation_alias=EnvVar.LLM_MEMORY_STORE)
    #: Rolling window (in output tokens) for the short-term tok/s rate.
    throughput_window_tokens: int = Field(default=1000, validation_alias=EnvVar.LLM_THROUGHPUT_WINDOW_TOKENS)
    delta_retry_enabled: bool = Field(default=True, validation_alias=EnvVar.LLM_DELTA_RETRY_ENABLED)
    batch_failure_threshold_ratio: float = Field(default=0.5, validation_alias=EnvVar.LLM_BATCH_FAILURE_THRESHOLD_RATIO)
    batch_failure_threshold_min: int = Field(default=5, validation_alias=EnvVar.LLM_BATCH_FAILURE_THRESHOLD_MIN)
    reasoning_effort: ReasoningEffort | None = Field(default=None, validation_alias=EnvVar.LLM_REASONING_EFFORT)
    context_manager: Annotated[ContextManagerPreset | bool, BeforeValidator(_coerce_context_manager)] = Field(
        default=ContextManagerPreset.DEFAULT,
        validation_alias=EnvVar.LLM_CONTEXT_MANAGER,
    )
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


class BudgetConfig(_KompendBaseModel):
    """Token budgets and structural limits for LLM operations.

    Sizing strategy:
    - Input chunks: ``clamp(context_window - reserve, chunk_size_min, chunk_size_max)``.
    - No output ceiling exists. ``max_tokens`` is never sent to a model: an
      answer is bounded by the context window minus its prompt, and the
      planners leave ``ContextWindow.answer_reserve`` free for it (see
      ``kompend.llm.context_window``).
    """

    verify_min_ngram: int = 3
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
    #: Warning threshold for the number of concept pages. Above it every
    #: candidate is compiled — the value never drops a candidate.
    max_concept_pages: int = Field(default=50, validation_alias=EnvVar.BUDGET_MAX_CONCEPT_PAGES)
    chunk_size_min: int = Field(default=2048, validation_alias=EnvVar.BUDGET_CHUNK_SIZE_MIN)
    chunk_size_max: int = Field(default=131072, validation_alias=EnvVar.BUDGET_CHUNK_SIZE_MAX)
    #: Characters per summary call. Longer content is split into several
    #: calls (slower, but no content is lost) — it is never truncated.
    summary_max_content_chars: int = 8000
    #: Characters per payload window of the agentic summarizer. One window
    #: costs one summarize call, so a bigger window means fewer calls (faster)
    #: — it never drops content.
    summary_window_chars: int = Field(default=2000, ge=1, validation_alias=EnvVar.BUDGET_SUMMARY_WINDOW_CHARS)
    summary_target_chars: int = 600
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
    summary_session_batch_size: int = Field(default=20, ge=1, validation_alias=EnvVar.BUDGET_SUMMARY_SESSION_BATCH_SIZE)
    summary_session_stall_retries: int = Field(default=1, ge=0, validation_alias=EnvVar.BUDGET_SUMMARY_SESSION_STALL_RETRIES)
    #: Characters served per ``read_source()`` call in the write-page
    #: synthesis. A longer source is read in several windows (the agent
    #: pages through it) — it is never shortened.
    capture_source_char_max: int = Field(default=4000, validation_alias=EnvVar.BUDGET_CAPTURE_SOURCE_CHAR_MAX)
    concept_merge_threshold: float = Field(default=0.82, validation_alias=EnvVar.BUDGET_CONCEPT_MERGE_THRESHOLD)
    concept_string_merge_threshold: float = Field(default=0.70, validation_alias=EnvVar.BUDGET_CONCEPT_STRING_MERGE_THRESHOLD)
    jaccard_min_token_len: int = Field(default=1, validation_alias=EnvVar.BUDGET_JACCARD_MIN_TOKEN_LEN)
    edge_min_title_len: int = Field(default=4, validation_alias=EnvVar.BUDGET_EDGE_MIN_TITLE_LEN)


class OutputConfig(_KompendBaseModel):
    """Configuration for output formatting and rendering.

    ``format`` defaults to ``markdown`` (an OKF v0.2-compliant vault).
    """

    format: OutputFormat = Field(default=OutputFormat.MARKDOWN, validation_alias=EnvVar.FORMAT)
    flavor: Annotated[Flavor | None, BeforeValidator(_coerce_flavor)] = Field(default=None, validation_alias=EnvVar.OUTPUT_FLAVOR)
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    rumdl_config: Path | None = Field(default=None, validation_alias=EnvVar.RUMDL_CONFIG)
    max_related_links: int = Field(default=15, validation_alias=EnvVar.OUTPUT_MAX_RELATED_LINKS)


class EmbeddingConfig(_KompendBaseModel):
    """Configuration for embedding models.

    Vector dimensionality is detected from the model at runtime via the
    embedding provider — no ``dim`` config value is needed.
    """

    model: str = Field(default="", validation_alias=EnvVar.LLM_EMBEDDING_MODEL)
    batch_size: int = Field(default=10, validation_alias=EnvVar.EMBEDDING_BATCH_SIZE)


# ── Other section configs ────────────────────────────────────────────────


class PathsConfig(_KompendBaseModel):
    """Vault path overrides (all optional — resolved from base_dir at runtime)."""

    wiki_dir: str | None = Field(default=None, validation_alias=EnvVar.WIKI_DIR)
    sources_dir: str = Field(default="sources", validation_alias=EnvVar.SOURCES_DIR)
    output_dir: str = Field(default="wiki", validation_alias=EnvVar.OUTPUT_DIR)
    cache_dir: str = Field(default=CACHE_DIR, validation_alias=EnvVar.CACHE_DIR)
    state_dir: str = Field(default=KOMP_DIR, validation_alias=EnvVar.STATE_DIR)
    recursive: bool = Field(
        default=True,
        validation_alias=EnvVar.PATHS_RECURSIVE,
        description=(
            "Discover sources in subdirectories of sources_dir. Set false for a flat "
            "vault: only files directly in sources_dir are indexed. "
            "CLI: --recursive/--no-recursive."
        ),
    )


class LintConfig(_KompendBaseModel):
    """Lint configuration. ``enabled=None`` means all structural rules."""

    enabled: list[str] | None = None
    fuzzy_threshold: float = Field(default=0.85, validation_alias=EnvVar.LINT_FUZZY_THRESHOLD)
    #: Per-call timeout for the LLM lint rules. They are long-context JSON
    #: requests (a batch of page summaries, or a whole page for clarity and
    #: factual checks), so a slow provider needs more than the compile default:
    #: a timed-out rule loses its findings instead of retrying the call.
    llm_timeout: float = Field(default=300.0, validation_alias=EnvVar.LINT_LLM_TIMEOUT)
    #: How much of a lint report reaches the terminal. ``quiet`` (default) prints
    #: the verdict and the path of the full report; ``summary`` adds one line per
    #: failing rule; ``full`` prints every finding and every passing rule. The
    #: full report is always written to ``.kompend/lint-log/``, and
    #: ``--verbose-lint`` / ``--quiet-lint`` override this per run.
    output: LintOutput = Field(default=LintOutput.QUIET, validation_alias=EnvVar.LINT_OUTPUT)
    model: str | None = Field(
        default=None,
        validation_alias=EnvVar.LINT_MODEL,
        description="Model override for LLM-powered lint steps (factual-consistency, clarity-score, ...). Falls back to compile.llm.default_model when unset.",
    )


class DedupConfig(_KompendBaseModel):
    """Deduplication configuration (opt-in by default)."""

    enabled: bool = Field(default=False, validation_alias=EnvVar.DEDUP_ENABLED)
    #: Comma-string (env var, legacy TOML) or array; both parse through
    #: :func:`parse_matchers`, which raises on an unknown name.
    matchers: Annotated[list[MatcherName], BeforeValidator(parse_matchers)] = Field(
        default_factory=lambda: list(DEFAULT_MATCHERS),
        validation_alias=EnvVar.DEDUP_MATCHERS,
    )
    threshold_identical: float = Field(default=0.95, validation_alias=EnvVar.DEDUP_THRESHOLD_IDENTICAL)
    threshold_similar: float = Field(default=0.80, validation_alias=EnvVar.DEDUP_THRESHOLD_SIMILAR)
    model: str = Field(default="", validation_alias=EnvVar.DEDUP_MODEL)
    minhash_lsh_threshold: float = Field(default=0.8, validation_alias=EnvVar.DEDUP_MINHASH_LSH_THRESHOLD)


class TrustConfig(_KompendBaseModel):
    """Privacy and trust configuration."""

    privacy_filter: bool = Field(default=True)
    user_tracking: bool = Field(default=True, validation_alias=EnvVar.USER_TRACKING)


class QueryConfig(_KompendBaseModel):
    """Query command configuration."""

    format: LlmDataFormat = LlmDataFormat.MARKDOWN


class SearchExpansionConfig(_KompendBaseModel):
    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_EXPANSION_ENABLED)
    variants: int = Field(default=3, validation_alias=EnvVar.SEARCH_EXPANSION_VARIANTS)


class SearchCragConfig(_KompendBaseModel):
    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_CRAG_ENABLED)
    threshold: float = Field(default=0.3, validation_alias=EnvVar.SEARCH_CRAG_THRESHOLD)


class SearchRerankerConfig(_KompendBaseModel):
    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_RERANKER_ENABLED)
    model: str = Field(default="", validation_alias=EnvVar.SEARCH_RERANKER_MODEL)
    top_k: int = Field(default=20, validation_alias=EnvVar.SEARCH_RERANKER_TOP_K)


class SearchHydeConfig(_KompendBaseModel):
    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_HYDE_ENABLED)


class SearchContextualConfig(_KompendBaseModel):
    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_CONTEXTUAL_ENABLED)
    model: str = Field(default="", validation_alias=EnvVar.SEARCH_CONTEXTUAL_MODEL)


class SearchAgenticConfig(_KompendBaseModel):
    """Opt-in agentic query path (agent searches the wiki via tools).

    Default off — the existing retrieve-then-synthesize path is unchanged.
    """

    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_AGENTIC_ENABLED)
    model: str = Field(default="", validation_alias=EnvVar.SEARCH_AGENTIC_MODEL)
    max_sub_questions: int = Field(default=5, validation_alias=EnvVar.SEARCH_AGENTIC_MAX_SUB_QUESTIONS)


class SearchSynthesisConfig(_KompendBaseModel):
    verbosity: Verbosity = Field(default=Verbosity.STANDARD, validation_alias=EnvVar.SEARCH_SYNTHESIS_VERBOSITY)


class AttachmentsConfig(_KompendBaseModel):
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


class SearchConfig(_KompendBaseModel):
    expansion: SearchExpansionConfig = Field(default_factory=SearchExpansionConfig)
    crag: SearchCragConfig = Field(default_factory=SearchCragConfig)
    reranker: SearchRerankerConfig = Field(default_factory=SearchRerankerConfig)
    hyde: SearchHydeConfig = Field(default_factory=SearchHydeConfig)
    contextual: SearchContextualConfig = Field(default_factory=SearchContextualConfig)
    agentic: SearchAgenticConfig = Field(default_factory=SearchAgenticConfig)
    synthesis: SearchSynthesisConfig = Field(default_factory=SearchSynthesisConfig)


class OrganizeConfig(_KompendBaseModel):
    """Configuration for ``kompend organize`` LLM-based file categorization."""

    max_depth: int = Field(default=3, validation_alias=EnvVar.ORGANIZE_MAX_DEPTH)
    min_subcategories: int = Field(default=2, validation_alias=EnvVar.ORGANIZE_MIN_SUBCATEGORIES)
    min_files_per_category: int = 2
    collapse_single_file: bool = Field(default=True, validation_alias=EnvVar.ORGANIZE_COLLAPSE_SINGLE_FILE)


class IgnoreConfig(_KompendBaseModel):
    patterns: list[str] = Field(default_factory=list)


class DisplayConfig(_KompendBaseModel):
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
    description_max_chars: int = Field(default=200, validation_alias=EnvVar.DISPLAY_DESCRIPTION_MAX_CHARS)
    cli_element_chars: int = Field(default=80, validation_alias=EnvVar.DISPLAY_CLI_ELEMENT_CHARS)
    tier_sample_size: int = Field(default=5, validation_alias=EnvVar.DISPLAY_TIER_SAMPLE_SIZE)
    cli_arg_preview: int = Field(default=4, validation_alias=EnvVar.DISPLAY_CLI_ARG_PREVIEW)
    log_error_chars: int = Field(default=500, validation_alias=EnvVar.DISPLAY_LOG_ERROR_CHARS)
    lint_feedback_chars: int = Field(default=200, validation_alias=EnvVar.DISPLAY_LINT_FEEDBACK_CHARS)
    memory_query_chars: int = Field(default=50, validation_alias=EnvVar.DISPLAY_MEMORY_QUERY_CHARS)
    mcp_element_chars: int = Field(default=500, validation_alias=EnvVar.DISPLAY_MCP_ELEMENT_CHARS)
    web_nodes: int = Field(default=10, validation_alias=EnvVar.DISPLAY_WEB_NODES)
    search_snippet_chars: int = Field(default=300, validation_alias=EnvVar.DISPLAY_SEARCH_SNIPPET_CHARS)


class NotesConfig(_KompendBaseModel):
    file: str = Field(default="NOTES.md", validation_alias=EnvVar.NOTES_FILE)


class IngestConfig(_KompendBaseModel):
    parsecraft_judge: ParsecraftJudge = Field(
        default=ParsecraftJudge.NONE,
        validation_alias=EnvVar.INGEST_PARSECRAFT_JUDGE,
        description=(
            "Judge-tier routing for the parsecraft backend. `none` (default) leaves the "
            "deterministic rule table in place — fast, but it cannot re-rank candidates per "
            "page on a model signal. `ollama` requires a local ollama daemon running a "
            "decision-capable model (capability `decision` in ollama's `details.capabilities`); "
            "the SDK rejects completion-only models with 400. `zen` and `typesafe-ai` are "
            "cloud providers that need credentials via `TYPESAFE_API_KEY` (or per-provider env)."
        ),
    )
    parsecraft_judge_model: str = Field(
        default="",
        validation_alias=EnvVar.INGEST_PARSECRAFT_JUDGE_MODEL,
        description=(
            "Provider-specific model token for the judge. For `ollama` the bare model name as "
            "ollama lists it (e.g. `parable-tinyjev`, `tev1`); the SDK validator rejects colon-suffixed "
            "ollama tag tokens like `parable-tinyjev:latest`. Empty string is rejected at parse "
            "time when `parsecraft_judge` is set to a non-`none` value — a provider without a model "
            "is a misconfiguration, not a sensible default."
        ),
    )
    parsecraft_preference: ParsecraftPreference = Field(
        default=ParsecraftPreference.BALANCED,
        validation_alias=EnvVar.INGEST_PARSECRAFT_PREFERENCE,
        description=(
            "Ranking axis the judge applies when several candidates tie on the constraint check. "
            "Maps directly to parsecraft's `RoutingPreference`: `speed` favours lower-VRAM backends "
            "(~native-fast first), `quality` favours the highest-VRAM strong backend, `balanced` is the "
            "default tiebreak — a per-page intent re-ranks over the eligible candidates regardless."
        ),
    )
    parsecraft_effort: ParsecraftEffort = Field(
        default=ParsecraftEffort.FLASH,
        validation_alias=EnvVar.INGEST_PARSECRAFT_EFFORT,
        description=(
            "Conversion effort for the MinerU sub-backend. Maps to parsecraft's `effort` backend option "
            "(`MineruEffort`). `flash` is the verified default (text PDFs, weight-free CPU work). `high` "
            "and `xhigh` engage the VLM path and emit equations, tables and figure captions at markedly "
            "higher recovery; they also load MinerU's VLM checkpoint and want a GPU. The value is forwarded "
            "to parsecraft as a per-backend constructor option and injected only when non-default, so an "
            "explicit `flash` does not churn the conversion cache."
        ),
    )
    parsecraft_allow_ocr: bool | None = Field(
        default=None,
        validation_alias=EnvVar.INGEST_PARSECRAFT_ALLOW_OCR,
        description=(
            "Whether parsecraft may route a page to an OCR sub-backend. Unset (`None`) keeps parsecraft's own "
            "derivation: OCR is permitted when an `ocr-*` extra is installed OR the host has a usable OCR engine. "
            "Set `false` to ban OCR outright — an all-native corpus should, since a blank or faint page otherwise "
            "pays a full GPU VLM pass — or `true` to permit it regardless of the derivation. The value is part "
            "of the parse-cache fingerprint, so changing it re-parses."
        ),
    )
    parsecraft_engines: list[str] = Field(
        default_factory=list,
        validation_alias=EnvVar.INGEST_PARSECRAFT_ENGINES,
        description=(
            "Restrict the external engines parsecraft may use, overriding the probed set. Empty (the default) "
            "keeps the probe's answer. Names are parsecraft's own engine ids (`marker`, `mineru`, …) — open "
            "capability data, so they are not a closed value domain here; the env form is a comma-separated "
            "list. The value is part of the parse-cache fingerprint, so changing it re-parses."
        ),
    )
    parsecraft_extract_vector_graphics: bool | None = Field(
        default=None,
        validation_alias=EnvVar.INGEST_PARSECRAFT_EXTRACT_VECTOR_GRAPHICS,
        description=(
            "Ask the liteparse backend to recover table structure from a page's vector graphics. Unset "
            "(`None`) leaves the option unset; `true` enables the pass, which populates "
            "`StructuredChunk.rows` so a table reaches kompend as `list[list[str]]` instead of Markdown. "
            "Pages without vector records still yield no TABLE chunks, and the option is inert on pages "
            "another backend serves."
        ),
    )
    parsecraft_preferred_backend: str | None = Field(
        default=None,
        validation_alias=EnvVar.INGEST_PARSECRAFT_PREFERRED_BACKEND,
        description=(
            "Rank a backend first where it is eligible (parsecraft's `preferred_backend`) — a preference, "
            "never a guarantee: a page the named backend cannot serve still falls back. Use it to pin a "
            "layout-aware backend for a table-heavy corpus, or to route around a defect in the leading "
            "backend without uninstalling anything. Unknown names are a usage error naming the registered "
            "backends; the value is part of the parse-cache fingerprint."
        ),
    )
    parsecraft_only_backend: str | None = Field(
        default=None,
        validation_alias=EnvVar.INGEST_PARSECRAFT_ONLY_BACKEND,
        description=(
            "Use only this backend — a hard pin (parsecraft's `only_backend`), unlike "
            "`parsecraft_preferred_backend`. Pages it cannot *cover* fail loudly instead of silently "
            "falling back; a blank page still degrades, because coverage and content are different "
            "things. The value is part of the parse-cache fingerprint."
        ),
    )
    #: Per-backend constructor options (`{backend: {key: value}}`) forwarded to
    #: parsecraft's `RoutingConstraints.backend_options`. Only non-default values
    #: are injected, since parsecraft hashes the dict verbatim into the
    #: conversion-cache key. Keys must be registered backend names
    #: (`mineru`, `liteparse`, `marker`, ...); values must be `str | int | float | bool`.
    parsecraft_backend_options: dict[str, dict[str, str | int | float | bool]] = Field(
        default_factory=dict,
        validation_alias=EnvVar.INGEST_PARSECRAFT_BACKEND_OPTIONS,
        description=(
            "Per-backend options (`{backend_name: {option_key: option_value}}`) forwarded to parsecraft's "
            "`RoutingConstraints.backend_options`. Only non-default values are injected, so an empty dict "
            "(`{}`) produces no `backend_options:` routing segment and the conversion cache stays stable. "
            'Example: ``{"mineru": {"effort": "high"}, "liteparse": {"extract_vector_graphics": true}}``. '
            "Keys must match parsecraft's registered backends; unknown backend names are a usage error at "
            "dispatch time. Values are coerced by the adapter's option-coercion site: `str -> bool` (real bools "
            "pass through unchanged), numeric strings -> int/float."
        ),
    )


class DownloaderConfig(_KompendBaseModel):
    undetected: bool = Field(default=True, validation_alias=EnvVar.DOWNLOADER_UNDETECTED)


class ExportConfig(_KompendBaseModel):
    """Where extracted knowledge lands in ``kompend sources export``."""

    annotations: ExportMode = Field(default=ExportMode.INLINE, validation_alias=EnvVar.EXPORT_ANNOTATIONS)
    summary: ExportMode = Field(default=ExportMode.INLINE, validation_alias=EnvVar.EXPORT_SUMMARY)


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


class ProviderModelConfig(_KompendBaseModel):
    """Per-model settings within a provider (``[providers.<name>.models."<model>"]``)."""

    context_window: int | None = Field(
        default=None, description="Context window length for this model (finest granularity, wins over provider-level and catalog values)"
    )


class ProviderEndpoint(_KompendBaseModel):
    """A user-defined provider endpoint.

    Defined under ``[providers.<name>]`` in ``kompend.toml`` (recommended:
    globally in ``~/.config/kompend/config.toml``, since providers carry
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
    api_type: LlmApiType = Field(
        default=LlmApiType.OPENAI_CHAT,
        description="Wire protocol the endpoint speaks: openai-chat (/v1/chat/completions), openai-responses (/v1/responses), or anthropic (/v1/messages)",
    )
    context_window: int | None = Field(
        default=None, description="Default context window length for all models on this provider (overrides auto-discovery; per-model entries win)"
    )
    models: dict[str, ProviderModelConfig] = Field(default_factory=dict, description="Per-model settings keyed by model name")
    max_connections: int | None = Field(
        default=None,
        description="Max parallel LLM connections to this provider (client-side). Clips compile.workers when set. None = no limit",
    )


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
    workers: int = Field(default=1, validation_alias=EnvVar.WORKERS)
    no_cache: bool = Field(default=False, validation_alias=EnvVar.NO_CACHE)
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


# ── Root config ──────────────────────────────────────────────────────────


class DecisionConfig(_KompendBaseModel):
    """Configuration for the optional decision-model sidecar (TypeSafe SystemOne).

    A decision model answers questions the pipeline cannot answer from types
    alone: "is this failure worth repeating?", later "which passage answers
    this?". It is optional - with ``enabled = false`` callers use the
    deterministic fallback in :mod:`kompend.llm.errors`.
    """

    enabled: bool = Field(default=False, validation_alias=EnvVar.DECISION_ENABLED)
    #: Full ``provider/model`` string (for example ``ollama/clef`` or
    #: ``typesafe/jev-latest``). Provider credentials come from the
    #: ``[providers.*]`` blocks - a decision model is not a special provider.
    model: str | None = Field(default=None, validation_alias=EnvVar.DECISION_MODEL)
    timeout: float = Field(default=10.0, validation_alias=EnvVar.DECISION_TIMEOUT)


class KompendConfig(_KompendBaseModel):
    """Root config model — single source of truth for DEFAULTS and _ENV_MAP.

    ``DEFAULTS = KompendConfig().model_dump()`` and ``_ENV_MAP`` is auto-generated
    by walking the model's ``validation_alias`` fields.
    """

    display: DisplayConfig = Field(default_factory=DisplayConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    compile: CompileConfig = Field(default_factory=CompileConfig)
    lint: LintConfig = Field(default_factory=LintConfig)
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


def _coerce_flavor(value: object) -> object:
    """Normalize a raw ``compile.output.flavor`` value for the schema.

    Lowercases, resolves aliases, then lets pydantic validate against the
    members; ``None`` passes through.
    """
    if isinstance(value, str) and value:
        return FLAVOR_ALIASES.get(value.strip().lower(), value.strip().lower())
    return value


def resolve_flavor(value: str) -> Flavor:
    """Resolve a programmatic flavor string to its member.

    Applies the same alias handling as the config schema; raises
    :class:`ValueError` for an unknown flavor.
    """
    normalized = FLAVOR_ALIASES.get(value.strip().lower(), value.strip().lower())
    return Flavor(normalized)
