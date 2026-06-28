"""``knox status`` — show per-wiki stats."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from knox.cli._sources import get_supported_extensions
from knox.core.config import find_project_config
from knox.core.paths import COMPILE_STATE_FILE, INDEX_FILE
from knox.core.tier_state import TierStateStore, classify_compile_status
from knox.logging import get_console


def collect_stats(
    sources_dir: Path | None = None,
    output_dir: Path | None = None,
    state_dir: Path | None = None,
) -> dict:
    """Collect wiki statistics. Gracefully handles missing dirs."""
    stats: dict = {
        "wiki_name": "KNOX Wiki",
        "config_path": None,
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

    # Article stats
    if output_dir and output_dir.exists():
        wiki_file = output_dir / "wiki.md"
        wiki_exists = wiki_file.exists()
        if wiki_exists:
            text = wiki_file.read_text(encoding="utf-8")
            word_count = len(text.split())
            stats["articles"]["count"] = text.count("\n#")  # approx page count
            stats["articles"]["total_words"] = word_count

            # Last compile timestamp (from mtime of wiki.md)
            mtime = wiki_file.stat().st_mtime
            stats["last_compile"] = datetime.fromtimestamp(mtime).isoformat()

        # Check for index.md / log.md (N11)
        index_file = output_dir / INDEX_FILE
        if index_file.exists():
            stats["articles"]["count"] = max(stats["articles"]["count"], 1)

        # Config path
        project_config = find_project_config()
        if project_config:
            stats["config_path"] = str(project_config)

    # Compile status from tier state
    state_dir_exists = bool(state_dir and state_dir.exists())
    tier_state = None
    error = ""
    if state_dir_exists:
        state_file = state_dir / COMPILE_STATE_FILE
        store = TierStateStore(state_file)
        try:
            tier_state = store.load()
        except (json.JSONDecodeError, ValueError) as exc:
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
    if stats["config_path"]:
        get_console().print(f"  Config:    {stats['config_path']}")
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
