"""``kompend status`` — show per-wiki stats."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from kompend.core.paths import (
    COMPILE_STATE_FILE,
    INDEX_FILE,
    KIND_FOLDERS,
    LOG_FILE,
    OVERVIEW_MD,
    QUALITY_FILE,
    REFERENCES_MD,
)
from kompend.core.pipeline.state import TierStateStore, classify_compile_status
from kompend.logging import get_console
from kompend.preprocessing.parsers._registry import get_supported_extensions

#: Root files of a compiled bundle that carry the bundle, not a page.
_BUNDLE_FILES: frozenset[str] = frozenset({INDEX_FILE, LOG_FILE, QUALITY_FILE, REFERENCES_MD, OVERVIEW_MD})
#: Folders the writer puts pages in — one page per file.
_PAGE_FOLDERS: frozenset[str] = frozenset(KIND_FOLDERS.values())


def _page_files(output_dir: Path) -> list[Path]:
    """The compiled pages of a bundle — one ``.md`` file per page.

    Pages live in the kind folders (``concepts/``, ``entities/``). The
    bundle root holds the reserved files (index, log, quality,
    references, overview); those describe the bundle, they are not pages.
    """
    pages: list[Path] = []
    for path in output_dir.rglob("*.md"):
        relative = path.relative_to(output_dir)
        if len(relative.parts) > 1:
            if relative.parts[0] in _PAGE_FOLDERS:
                pages.append(path)
        elif path.name not in _BUNDLE_FILES:
            pages.append(path)
    return pages


def collect_stats(
    sources_dir: Path | None = None,
    output_dir: Path | None = None,
    state_dir: Path | None = None,
    config_file: Path | None = None,
) -> dict:
    """Collect wiki statistics. Gracefully handles missing dirs."""
    stats: dict = {
        "wiki_name": "kompend Wiki",
        "config_file": None,
        "sources": {"count": 0, "total_size_bytes": 0},
        "articles": {"count": 0, "total_words": 0},
        "last_compile": None,
        "tier_distribution": None,
        "compile_status": None,
        "health_score": None,
    }

    # Source stats
    if sources_dir and sources_dir.exists():
        source_files: list[Path] = []
        for ext in get_supported_extensions():
            source_files.extend(sources_dir.rglob(f"*{ext}"))
        stats["sources"]["count"] = len(source_files)
        stats["sources"]["total_size_bytes"] = sum(f.stat().st_size for f in source_files if f.is_file())

    wiki_exists = False

    # Article stats — one .md file per page
    if output_dir and output_dir.exists():
        page_files = _page_files(output_dir)
        wiki_exists = bool(page_files)
        if page_files:
            total_words = 0
            latest_mtime = 0.0
            for pf in page_files:
                total_words += len(pf.read_text(encoding="utf-8").split())
                latest_mtime = max(latest_mtime, pf.stat().st_mtime)
            stats["articles"]["count"] = len(page_files)
            stats["articles"]["total_words"] = total_words
            if latest_mtime:
                stats["last_compile"] = datetime.fromtimestamp(latest_mtime).isoformat()

        # Check for index.md / log.md (N11)
        index_file = output_dir / INDEX_FILE
        if index_file.exists():
            stats["articles"]["count"] = max(stats["articles"]["count"], 1)

    # Config path — the vault's own config, not one found in the cwd.
    if config_file is not None and config_file.exists():
        stats["config_file"] = str(config_file)

    # Compile status from tier state
    state_dir_exists = bool(state_dir and state_dir.exists())
    tier_state = None
    error = ""
    if state_dir is not None and state_dir.exists():
        state_file = state_dir / COMPILE_STATE_FILE
        store = TierStateStore(state_file)
        try:
            tier_state = store.load()
        except (json.JSONDecodeError, ValueError, sqlite3.Error) as exc:
            error = str(exc)

    info = classify_compile_status(
        state_dir_exists=state_dir_exists,
        sources_count=stats["sources"]["count"],
        tier_state=tier_state,
        wiki_exists=wiki_exists,
        error=error,
    )
    stats["compile_status"] = {
        "status": info.status.value,
        "label": info.label(),
        "total": info.total,
        "indexed": info.indexed,
        "compiled": info.compiled,
        "error": info.error,
    }

    if tier_state and tier_state.entries:
        stats["tier_distribution"] = tier_state.tier_distribution()

    return stats


def print_status(stats: dict, json_output: bool = False) -> int:
    """Print status. Returns exit code 0."""
    if json_output:
        json.dump(stats, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    get_console().print(f"  Wiki:      {stats['wiki_name']}")
    if stats["config_file"]:
        get_console().print(f"  Config:    {stats['config_file']}")
    else:
        get_console().print("  Config:    (none — using defaults)")

    src = stats["sources"]
    size_mb = src["total_size_bytes"] / (1024 * 1024)
    get_console().print(f"  Sources:   {src['count']} file(s)  ({size_mb:.1f} MB)")

    arts = stats["articles"]
    get_console().print(f"  Articles:  {arts['count']} page(s)  ({arts['total_words']:,} words)")

    if stats["last_compile"]:
        get_console().print(f"  Compiled:  {stats['last_compile']}")
    else:
        get_console().print("  Compiled:  (never)")

    cs = stats.get("compile_status")
    if cs:
        get_console().print(f"  Status:    {cs['label']}")

    if stats["tier_distribution"]:
        get_console().print(f"  Tiers:     {stats['tier_distribution']}")

    if stats["health_score"] is not None:
        get_console().print(f"  Health:    {stats['health_score']}/100")

    return 0
