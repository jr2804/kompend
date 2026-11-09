"""Config schema models.

The public surface is re-exported here; import from
:mod:`kompend.core.config.models`, not from a section module.
"""

from __future__ import annotations

from kompend.core.config.enums import (
    Compiler,
    ContextManagerPreset,
    ExtractionStrategy,
    Flavor,
    ImageMode,
    LlmApiType,
    LlmMemoryStore,
    MatcherName,
    OutputFormat,
    ReasoningEffort,
    RerankerBackend,
    Tier,
    TierInfo,
)
from kompend.core.config.models.attachments import AttachmentsConfig
from kompend.core.config.models.compile import (
    BudgetConfig,
    CompileConfig,
    ExtractionConfig,
    SkillConfig,
)
from kompend.core.config.models.decision import DecisionConfig
from kompend.core.config.models.eval import LintConfig
from kompend.core.config.models.ingest import (
    DownloaderConfig,
    IgnoreConfig,
    IngestConfig,
    OrganizeConfig,
)
from kompend.core.config.models.llm import (
    CompactionConfig,
    EmbeddingConfig,
    LLMConfig,
    ProviderEndpoint,
    ProviderModelConfig,
    RlmConfig,
)
from kompend.core.config.models.memory import (
    DEFAULT_MATCHERS,
    DedupConfig,
    parse_matchers,
)
from kompend.core.config.models.output import (
    FLAVOR_ALIASES,
    DisplayConfig,
    ExportConfig,
    OutputConfig,
    resolve_flavor,
)
from kompend.core.config.models.root import (
    DEFAULTS,
    KompendConfig,
    LoggingConfig,
    NotesConfig,
    PathsConfig,
    TrustConfig,
)
from kompend.core.config.models.search import (
    QueryConfig,
    SearchAgenticConfig,
    SearchConfig,
    SearchContextualConfig,
    SearchCragConfig,
    SearchExpansionConfig,
    SearchHydeConfig,
    SearchRerankerConfig,
    SearchSynthesisConfig,
)
from kompend.core.vocab import ExportMode, LintOutput, LlmDataFormat, Verbosity

__all__ = [
    "DEFAULTS",
    "DEFAULT_MATCHERS",
    "FLAVOR_ALIASES",
    "AttachmentsConfig",
    "BudgetConfig",
    "CompactionConfig",
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
    "LoggingConfig",
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
