"""Structural protocol for the pipeline object passed to stage functions.

Stage modules (``_parse.py``, ``_index.py``, ``_compile.py``,
``_resolve.py``) receive the orchestrator as a parameter but cannot import
:class:`WikiPipeline` directly (circular import — ``_pipeline.py`` imports
the stages). This Protocol breaks the cycle: stages type-hint against the
abstraction, ``WikiPipeline`` satisfies it structurally.

Only the attributes and methods that stages actually access are declared.
All members are public — stages are the interface's consumers, so nothing
here is private.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from knox.chunking.hierarchical import HierarchicalChunker
from knox.chunking.summarizer import NodeSummarizer
from knox.compilation.base import BaseCompiler
from knox.core.cache.store import CacheStore
from knox.core.models import ChunkTree
from knox.core.models.attachments import AttachmentManifest
from knox.core.pipeline.config import PipelineConfig
from knox.core.pipeline.state import TierState, TierStateStore
from knox.core.resolution.xref_checker import CrossReferenceResult
from knox.output.base import BaseOutputWriter
from knox.preprocessing.downloaders.base import BaseDownloader
from knox.preprocessing.parsers.base import ParserRouter
from knox.resolution.base import BaseResolver
from knox.types import MetadataValue

__all__ = ["PipelineRuntime", "ProgressCallback"]

#: Type alias for the pipeline progress-event callback.
#: Parameters: (phase, current, total, detail).
ProgressCallback = Callable[[str, int, int, str], None]

class PipelineRuntime(Protocol):
    """Structural interface that stage functions depend on.

    :class:`WikiPipeline` satisfies this structurally; no runtime
    registration is needed.
    """

    # ── Configuration & paths ──────────────────────────────────────────
    config: PipelineConfig
    base_directory: Path

    # ── Processing components ──────────────────────────────────────────
    router: ParserRouter | None
    chunker: HierarchicalChunker | None
    resolver: BaseResolver | None
    compiler: BaseCompiler | None
    output_writer: BaseOutputWriter | None
    summarizer: NodeSummarizer | None
    downloader: BaseDownloader | None

    # ── State & cache ──────────────────────────────────────────────────
    state_store: TierStateStore
    cache_store: CacheStore
    last_chunk_count: int
    xref_result: CrossReferenceResult | None

    @property
    def state(self) -> TierState: ...

    @property
    def attachments(self) -> AttachmentManifest: ...

    # ── Methods used by stages ─────────────────────────────────────────
    def emit(self, phase: str, current: int, total: int, detail: str = "") -> None: ...

    def rel_file(self, file: Path) -> str: ...

    def abs_file(self, file_str: str) -> Path: ...

    def cache_directory(self) -> Path: ...

    def dedup_store_file(self) -> Path: ...

    def normalize_metadata_paths(
        self, metadata: dict[str, MetadataValue]
    ) -> dict[str, MetadataValue]: ...

    def apply_tags_to_tree(self, tree: ChunkTree, source_file: Path) -> None: ...

    def iter_sources(self) -> list[Path]: ...
