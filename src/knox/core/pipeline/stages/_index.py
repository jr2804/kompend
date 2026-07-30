"""Tier 0 (index) orchestration helpers."""

from __future__ import annotations

from typing import Any

from knox.core.pipeline.config import Tier
from knox.preprocessing.parsers._registry import get_all_extensions


def iter_sources(pipeline: Any) -> list:
    """Yield source files to process.

    Always scans ``config.source_dir`` for supported file types.
    The ``config.input_files`` list is only used by the ingest
    command to copy files into the source directory — it does
    *not* scope which files the pipeline processes.

    Excludes the URL list file — it is processed via
    ``--url-file``, not a document to parse.
    """
    extensions = get_all_extensions()
    sources = []
    for f in pipeline.config.source_dir.rglob("*"):
        if f.suffix.lower() in extensions:
            sources.append(f)
    return sources


def run_index_tier(pipeline: Any) -> None:
    """Tier 0: download online sources and record state for local files."""
    if pipeline.downloader and pipeline.config.online_sources:
        for url in pipeline.config.online_sources:
            pipeline.downloader.download(url, pipeline.config.source_dir)


def detect_resume_tier(pipeline: Any) -> str | None:
    """Check compile_state.json for previous progress.

    Returns the tier to resume at, or ``None`` if no previous run
    exists or all sources are at tier 0 (no meaningful progress).
    """
    state = pipeline.state
    if not state.entries:
        return None
    min_tier = min(e.current_tier for e in state.entries.values())
    if min_tier < 1:
        return None
    resume_level = min_tier + 1
    for ti in Tier.all_tiers():
        if ti.level == resume_level:
            return ti.tier
    return None
