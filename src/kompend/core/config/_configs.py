"""Config enum vocabulary.

The section models live in :mod:`kompend.core.config.models`; this module defines
the enums and forwards the historical model imports lazily so existing call
sites keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import TYPE_CHECKING

from kompend.core.vocab import ExportMode, LintOutput, LlmDataFormat, Verbosity

if TYPE_CHECKING:
    # Static mirror of the lazy forwards below: type checkers resolve the real
    # model classes for the historical import path without a runtime cycle.
    from kompend.core.config.models import (
        DEFAULT_MATCHERS,
        DEFAULTS,
        FLAVOR_ALIASES,
        AttachmentsConfig,
        BudgetConfig,
        CompileConfig,
        DecisionConfig,
        DedupConfig,
        DisplayConfig,
        DownloaderConfig,
        EmbeddingConfig,
        ExportConfig,
        ExtractionConfig,
        IgnoreConfig,
        IngestConfig,
        KompendConfig,
        LintConfig,
        LLMConfig,
        NotesConfig,
        OrganizeConfig,
        OutputConfig,
        PathsConfig,
        ProviderEndpoint,
        ProviderModelConfig,
        QueryConfig,
        RlmConfig,
        SearchAgenticConfig,
        SearchConfig,
        SearchContextualConfig,
        SearchCragConfig,
        SearchExpansionConfig,
        SearchHydeConfig,
        SearchRerankerConfig,
        SearchSynthesisConfig,
        SkillConfig,
        TrustConfig,
        parse_matchers,
        resolve_flavor,
    )

__all__ = [
    "DEFAULTS",
    "DEFAULT_MATCHERS",
    "FLAVOR_ALIASES",
    "AttachmentsConfig",
    "BudgetConfig",
    "CompileConfig",
    "Compiler",
    "ContextManagerPreset",
    "DecisionConfig",
    "DedupConfig",
    "DisplayConfig",
    "DownloaderConfig",
    "EmbeddingConfig",
    "ExportConfig",
    "ExportMode",
    "ExtractionConfig",
    "ExtractionStrategy",
    "Flavor",
    "IgnoreConfig",
    "ImageMode",
    "IngestConfig",
    "KompendConfig",
    "LLMConfig",
    "LintConfig",
    "LintOutput",
    "LlmApiType",
    "LlmDataFormat",
    "LlmMemoryStore",
    "MatcherName",
    "NotesConfig",
    "OrganizeConfig",
    "OutputConfig",
    "OutputFormat",
    "PathsConfig",
    "ProviderEndpoint",
    "ProviderModelConfig",
    "QueryConfig",
    "ReasoningEffort",
    "RerankerBackend",
    "RlmConfig",
    "SearchAgenticConfig",
    "SearchConfig",
    "SearchContextualConfig",
    "SearchCragConfig",
    "SearchExpansionConfig",
    "SearchHydeConfig",
    "SearchRerankerConfig",
    "SearchSynthesisConfig",
    "SkillConfig",
    "Tier",
    "TierInfo",
    "TrustConfig",
    "Verbosity",
    "parse_matchers",
    "resolve_flavor",
]


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


class ImageMode(StrEnum):
    """Image handling mode for document parsing.

    - ``FILE``: save images as files (default)
    - ``BASE64``: embed images as base64-encoded data URIs
    - ``NONE``: extract no images — the text layer only
    """

    FILE = "file"
    BASE64 = "base64"
    NONE = "none"


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


class RerankerBackend(StrEnum):
    """How the re-rank stage scores the retrieved candidates.

    - ``BI_ENCODER``: re-score with the vector store's own embedding model.
      The candidates were ranked by that model already, so this mostly
      reorders them.
    - ``FLASHRANK``: re-score with a cross-encoder that reads the query and
      the passage together. Needs the ``rerank`` extra.
    - ``NONE``: skip the stage.
    """

    BI_ENCODER = "bi-encoder"
    FLASHRANK = "flashrank"
    NONE = "none"


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


def __getattr__(name: str) -> object:
    """Resolve a schema model lazily from the ``models`` package.

    The enums above are authoritative here; the section models live in
    :mod:`kompend.core.config.models`. Resolving them on first access keeps
    this historical import path working without a module-level cycle.
    """
    if name.startswith("__"):
        raise AttributeError(name)
    try:
        models = _models
    except NameError:
        raise AttributeError(name) from None
    try:
        return getattr(models, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


# Bottom import: the model package imports these enums, so it cannot be a
# top-level import here without a cycle.
from kompend.core.config import models as _models  # noqa: E402
