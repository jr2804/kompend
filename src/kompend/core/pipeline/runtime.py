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

from kompend.chunking.contextual import ContextualEnricher
from kompend.chunking.hierarchical import HierarchicalChunker
from kompend.chunking.summarizer import NodeSummarizer
from kompend.compilation.base import BaseCompiler
from kompend.core.cache.store import CacheStore
from kompend.core.models import ChunkTree
from kompend.core.pipeline.config import PipelineConfig
from kompend.core.pipeline.state import TierState, TierStateStore
from kompend.core.resolution.xref_checker import CrossReferenceResult
from kompend.output.base import BaseOutputWriter
from kompend.preprocessing.downloaders.base import BaseDownloader
from kompend.preprocessing.parsers.base import ParserRouter
from kompend.resolution.base import BaseResolver
from kompend.types import MetadataValue

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
    enricher: ContextualEnricher | None
    downloader: BaseDownloader | None

    # ── State & cache ──────────────────────────────────────────────────
    state_store: TierStateStore
    cache_store: CacheStore
    last_chunk_count: int
    xref_result: CrossReferenceResult | None

    @property
    def state(self) -> TierState: ...

    def emit(self, phase: str, current: int, total: int, detail: str = "") -> None: ...

    def rel_file(self, file: Path) -> str: ...

    def abs_file(self, file_str: str) -> Path: ...

    def cache_directory(self) -> Path: ...

    def dedup_store_file(self) -> Path: ...

    def normalize_metadata_paths(self, metadata: dict[str, MetadataValue]) -> dict[str, MetadataValue]: ...

    def apply_tags_to_tree(self, tree: ChunkTree, source_file: Path) -> None: ...

    def iter_sources(self) -> list[Path]: ...
