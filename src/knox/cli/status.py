"""``knox status`` — show per-wiki stats."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from knox.cli._sources import SUPPORTED_EXTENSIONS


def collect_stats(
    sources_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Collect wiki statistics. Gracefully handles missing dirs."""
    stats: dict = {
        "wiki_name": "KNOX Wiki",
        "config_path": None,
        "sources": {"count": 0, "total_size_bytes": 0},
        "articles": {"count": 0, "total_words": 0},
        "last_compile": None,
        "tier_distribution": None,
        "health_score": None,
    }

    # Source stats
    if sources_dir and sources_dir.exists():
        source_files: list[Path] = []
        for ext in SUPPORTED_EXTENSIONS:
            source_files.extend(sources_dir.rglob(f"*{ext}"))
        stats["sources"]["count"] = len(source_files)
        stats["sources"]["total_size_bytes"] = sum(
            f.stat().st_size for f in source_files if f.is_file()
        )

    # Article stats
    if output_dir and output_dir.exists():
        wiki_file = output_dir / "wiki.md"
        if wiki_file.exists():
            text = wiki_file.read_text(encoding="utf-8")
            word_count = len(text.split())
            stats["articles"]["count"] = text.count("\n#")  # approx page count
            stats["articles"]["total_words"] = word_count

        # Last compile timestamp (from mtime of wiki.md)
        if wiki_file.exists():
            mtime = wiki_file.stat().st_mtime
            stats["last_compile"] = datetime.fromtimestamp(mtime).isoformat()

        # Check for index.md / log.md (N11)
        index_file = output_dir / "index.md"
        if index_file.exists():
            stats["articles"]["count"] = max(stats["articles"]["count"], 1)

        # Config path
        knox_config = Path.cwd() / ".knox" / "config.yaml"
        if knox_config.exists():
            stats["config_path"] = str(knox_config)

    return stats


def print_status(stats: dict, json_output: bool = False) -> int:
    """Print status. Returns exit code 0."""
    if json_output:
        json.dump(stats, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    print(f"  Wiki:      {stats['wiki_name']}")
    if stats["config_path"]:
        print(f"  Config:    {stats['config_path']}")
    else:
        print("  Config:    (none — using defaults)")

    src = stats["sources"]
    size_mb = src["total_size_bytes"] / (1024 * 1024)
    print(f"  Sources:   {src['count']} file(s)  ({size_mb:.1f} MB)")

    arts = stats["articles"]
    print(f"  Articles:  {arts['count']} page(s)  ({arts['total_words']:,} words)")

    if stats["last_compile"]:
        print(f"  Compiled:  {stats['last_compile']}")
    else:
        print("  Compiled:  (never)")

    if stats["tier_distribution"]:
        print(f"  Tiers:     {stats['tier_distribution']}")

    if stats["health_score"] is not None:
        print(f"  Health:    {stats['health_score']}/100")

    return 0
