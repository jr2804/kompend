"""Root config model and the built-in DEFAULTS mapping."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from kompend.core.config.models._base import _KompendBaseModel
from kompend.core.config.models.attachments import AttachmentsConfig
from kompend.core.config.models.compile import CompileConfig
from kompend.core.config.models.decision import DecisionConfig
from kompend.core.config.models.eval import LintConfig
from kompend.core.config.models.ingest import DownloaderConfig, IgnoreConfig, IngestConfig, OrganizeConfig
from kompend.core.config.models.llm import ProviderEndpoint
from kompend.core.config.models.memory import DedupConfig
from kompend.core.config.models.output import DisplayConfig, ExportConfig
from kompend.core.config.models.search import QueryConfig, SearchConfig
from kompend.core.keys import EnvVar
from kompend.core.paths import CACHE_DIR, KOMP_DIR


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


class NotesConfig(_KompendBaseModel):
    file: str = Field(default="NOTES.md", validation_alias=EnvVar.NOTES_FILE)


class TrustConfig(_KompendBaseModel):
    """Privacy and trust configuration."""

    privacy_filter: bool = Field(default=True)
    user_tracking: bool = Field(default=True, validation_alias=EnvVar.USER_TRACKING)


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


DEFAULTS: dict[str, Any] = KompendConfig().model_dump()
