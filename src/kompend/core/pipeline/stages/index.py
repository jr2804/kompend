"""Tier 0 (index) orchestration helpers."""

from __future__ import annotations

from pathlib import Path

from kompend.core.ignore import is_special_excluded, load_gitignore, matches_any
from kompend.core.pipeline.config import Tier
from kompend.core.pipeline.runtime import PipelineRuntime
from kompend.logging import get_logger
from kompend.preprocessing.parsers.extensions import supported_extensions

logger = get_logger(__name__)


def iter_sources(pipeline: PipelineRuntime) -> list[Path]:
    """Yield source files to process.

    Always scans ``config.source_dir`` for supported file types - recursively
    unless ``config.recursive`` is false (flat vault: only files directly in
    ``source_dir``).
    The ``config.input_files`` list is only used by the ingest
    command to copy files into the source directory - it does
    *not* scope which files the pipeline processes.

    ``config.ignore_patterns`` (``[ignore] patterns``) is applied here: a file
    matching one of them is never a source, whether the run was started with
    explicit file arguments or with a bare vault scan. ``config.respect_gitignore``
    adds the source dir's ``.gitignore`` patterns on top.

    Excludes the URL list file - it is processed via
    ``--url-file``, not a document to parse.
    """
    extensions = supported_extensions()
    base_dir = pipeline.config.source_dir
    patterns = list(pipeline.config.ignore_patterns or [])
    if pipeline.config.respect_gitignore:
        patterns.extend(load_gitignore(base_dir))

    sources = []
    skipped = 0
    pattern = "**/*" if pipeline.config.recursive else "*"
    for f in base_dir.glob(pattern):
        if is_special_excluded(f):
            continue
        if f.suffix.lower() not in extensions:
            continue
        if patterns and matches_any(f, patterns, base_dir=base_dir):
            skipped += 1
            continue
        sources.append(f)
    if skipped:
        logger.info("Skipped %d file(s) matching ignore patterns", skipped)
    return sources


def run_index_tier(pipeline: PipelineRuntime) -> None:
    """Tier 0: download online sources and record state for local files."""
    if pipeline.downloader and pipeline.config.online_sources:
        for url in pipeline.config.online_sources:
            pipeline.downloader.download(url, pipeline.config.source_dir)


def detect_resume_tier(pipeline: PipelineRuntime) -> str | None:
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
