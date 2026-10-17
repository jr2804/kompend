"""kompend CLI argument types organized by command/domain.

This module re-exports all argument types from submodules to maintain
backward compatibility with existing imports. Use specific submodule imports
for better organization:

    from kompend.cli.args.common import ConfigPath, WikiDir
    from kompend.cli.args.query import QueryText, QueryTopK
    from kompend.cli.args.compile import OnlineUrls
"""

from __future__ import annotations

# Common/global arguments
from .common import (
    APP_NAME_UPPERCASE,
    DEFAULT_LLM_FALLBACK_MAX_RETRIES,
    DEFAULT_START_AT_TIER,
    CliSet,
    ConfigPath,
    JsonOutput,
    NoUserTracking,
    Progress,
    Recursive,
    Verbose,
    Version,
    WikiDir,
)

# Compile command
from .compile import (
    CompileLint,
    CompileLintAll,
    EmbedModel,
    Frontmatter,
    Ignore,
    IndexStore,
    LlmDefaultModel,
    LlmFallbackChain,
    LlmFallbackMaxRetries,
    OnlineUrls,
    SynthesizeFlag,
    UrlFile,
    VlmModel,
)

# Config command
from .config import (
    ConfigCheckPaths,
    ConfigGlobalOpt,
    ConfigKeyArg,
    ConfigNoMerge,
    ConfigValueArg,
)

# Init command
from .init import InitBare, InitConfig, InitForce, InitGit, InitKeepCache, InitTarget

# Knowledge commands (attachments, references, capture, pack, organize, agents)
from .knowledge import (
    WRITE_SOURCES_DEFAULT,
    AttachmentAll,
    AttachmentEmbed,
    AttachmentFile,
    AttachmentIngestible,
    AttachmentListAll,
    AttachmentListSource,
    AttachmentPath,
    AttachmentPathOptional,
    AttachmentSource,
    CaptureText,
    ConsolidateDryRun,
    ConsolidateNoArchive,
    DistillNoArchive,
    IngestTarget,
    LearnText,
    OrganizeCollapseSingle,
    OrganizeDryRun,
    OrganizeMaxDepth,
    OrganizeMinSubcategories,
    PackName,
    ReferencesExportOutput,
    ReferencesExportSource,
    ReferencesFetchDryRun,
    ReferencesFetchSource,
    ReferencesListSource,
    SourceAttrs,
    SourceTags,
    WritePageTitle,
    WriteSourceFiles,
)

# Memory command
from .memory import (
    MemoryAnswer,
    MemoryFeedback,
    MemoryLimit,
    MemoryQuery,
    MemoryTopK,
)
from .parse import ParseFiles, PrintElements

# Pipeline/tiered compilation
from .pipeline import (
    ExportAll,
    ExportAnnotations,
    ExportInline,
    ExportSourceId,
    ExportSummary,
    ForceTier,
    IngestFiles,
    InPlaceFlag,
    JsonProgress,
    ListTiers,
    MemoryEnabled,
    NoCacheFlag,
    StartAtTier,
    Workers,
)

# Quality commands (lint, review, provenance, doctor, status)
from .quality import (
    BypassReview,
    LintDryRun,
    LintFix,
    LintFixDestructive,
    LintFormatCheck,
    LintGapsFix,
    LintHistoryLast,
    LintHistoryTrend,
    LintLlm,
    LintPruneAgeDays,
    LintPruneKeep,
    LintRule,
    LintSemantic,
    LogLast,
    LogWrite,
    NoPrivacy,
    Private,
    Provenance,
    ProvenanceSlugs,
    ProvenanceWithContent,
    ReviewAction,
    ReviewAll,
    ReviewMode,
    ReviewReason,
    ReviewSlug,
    Verify,
)

# Query command
from .query import (
    QueryAdaptive,
    QueryFormatOption,
    QueryNoClassify,
    QueryRaw,
    QueryRemember,
    QuerySearchMode,
    QueryText,
    QueryTopK,
    QueryVerbosity,
    SearchMode,
)

# Serving commands (serve, mcp, agent)
from .serving import (
    AgentMcpPort,
    AgentsAction,
    AgentsName,
    AgentsTask,
    ServeHost,
    ServeOpen,
    ServePort,
)

__all__ = [
    # Common
    "APP_NAME_UPPERCASE",
    "DEFAULT_LLM_FALLBACK_MAX_RETRIES",
    "DEFAULT_START_AT_TIER",
    "WRITE_SOURCES_DEFAULT",
    "AgentMcpPort",
    "AgentsAction",
    "AgentsName",
    "AgentsTask",
    "AttachmentAll",
    "AttachmentEmbed",
    "AttachmentFile",
    "AttachmentIngestible",
    "AttachmentListAll",
    "AttachmentListSource",
    "AttachmentPath",
    "AttachmentPathOptional",
    "AttachmentSource",
    # Other
    "BypassReview",
    # Knowledge
    "CaptureText",
    "CliSet",
    # Compile
    "CompileLint",
    "CompileLintAll",
    # Config
    "ConfigCheckPaths",
    "ConfigGlobalOpt",
    "ConfigKeyArg",
    "ConfigNoMerge",
    "ConfigPath",
    "ConfigValueArg",
    "ConsolidateDryRun",
    "ConsolidateNoArchive",
    "DistillNoArchive",
    "EmbedModel",
    # Pipeline
    "ExportAll",
    "ExportAnnotations",
    "ExportInline",
    "ExportSourceId",
    "ExportSummary",
    "ForceTier",
    "Frontmatter",
    "Ignore",
    "InPlaceFlag",
    "IndexStore",
    "IngestFiles",
    "IngestTarget",
    # Init
    "InitBare",
    "InitConfig",
    "InitForce",
    "InitGit",
    "InitKeepCache",
    "InitTarget",
    "JsonOutput",
    "JsonProgress",
    "LearnText",
    "LintDryRun",
    "LintFix",
    "LintFixDestructive",
    "LintFormatCheck",
    "LintGapsFix",
    "LintHistoryLast",
    "LintHistoryTrend",
    "LintLlm",
    "LintPruneAgeDays",
    "LintPruneKeep",
    "LintRule",
    "LintSemantic",
    "ListTiers",
    "LlmDefaultModel",
    "LlmFallbackChain",
    "LlmFallbackMaxRetries",
    "LogLast",
    "LogWrite",
    # Memory
    "MemoryAnswer",
    "MemoryEnabled",
    "MemoryFeedback",
    "MemoryLimit",
    "MemoryQuery",
    "MemoryTopK",
    "NoCacheFlag",
    "NoPrivacy",
    "NoUserTracking",
    "OnlineUrls",
    "OrganizeCollapseSingle",
    "OrganizeDryRun",
    "OrganizeMaxDepth",
    "OrganizeMinSubcategories",
    "PackName",
    "ParseFiles",
    "PrintElements",
    "Private",
    "Progress",
    "Provenance",
    "ProvenanceSlugs",
    "ProvenanceWithContent",
    # Query
    "QueryAdaptive",
    "QueryFormatOption",
    "QueryNoClassify",
    "QueryRaw",
    "QueryRemember",
    "QuerySearchMode",
    "QueryText",
    "QueryTopK",
    "QueryVerbosity",
    "Recursive",
    "ReferencesExportOutput",
    "ReferencesExportSource",
    "ReferencesFetchDryRun",
    "ReferencesFetchSource",
    "ReferencesListSource",
    "ReviewAction",
    "ReviewAll",
    # Serving
    "ReviewMode",
    "ReviewReason",
    "ReviewSlug",
    "SearchMode",
    "ServeHost",
    "ServeOpen",
    "ServePort",
    "SourceAttrs",
    "SourceTags",
    "StartAtTier",
    "SynthesizeFlag",
    "UrlFile",
    "Verbose",
    "Verify",
    "Version",
    "VlmModel",
    "WikiDir",
    "Workers",
    "WritePageTitle",
    "WriteSourceFiles",
]
