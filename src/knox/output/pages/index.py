"""Generate ``index.md`` — TOC + stats for the compiled wiki."""

from __future__ import annotations

from pathlib import Path

from knox.core.config import DEFAULTS
from knox.core.config.keys import DISPLAY_WIKI_TOPICS
from knox.core.models import CompiledWiki, MetadataField, WikiField
from knox.core.paths import INDEX_FILE
from knox.core.utils.time import display_local


def write_index(
    wiki: CompiledWiki,
    output_dir: Path,
    *,
    base_dir: Path | None = None,
) -> dict[str, Path]:
    """Write ``index.md`` for a compiled wiki.

    The human-readable pipeline log is no longer auto-written; regenerate it
    on demand with ``knox log`` (or ``knox log --write``).

    Returns a dict of ``{name: path}`` for generated files.
    """
    resolved_base = base_dir if base_dir is not None else output_dir
    resolved_base.mkdir(parents=True, exist_ok=True)

    # index.md — entry point at base_dir level
    index_file = resolved_base / INDEX_FILE
    index_content = generate_index(wiki)
    index_file.write_text(index_content, encoding="utf-8")

    return {"index": index_file}


def generate_index(wiki: CompiledWiki) -> str:
    """Generate ``index.md`` — TOC + stats for the compiled wiki."""
    lines: list[str] = ["# Wiki Index\n"]

    # Stats summary
    source_count = wiki.metadata.get(MetadataField.SOURCE_COUNT, 0)
    article_count = len(wiki.nodes)
    total_words = sum(len(n.content.split()) for n in wiki.nodes)
    last_compile = display_local("%Y-%m-%d %H:%M")
    duration = wiki.metadata.get(WikiField.DURATION_SECONDS)

    lines.append("## Summary\n")
    lines.append(f"- **Sources:** {source_count}")
    lines.append(f"- **Articles:** {article_count}")
    lines.append(f"- **Total words:** {total_words:,}")
    lines.append(f"- **Last compile:** {last_compile}")
    if duration is not None:
        lines.append(f"- **Duration:** {duration:.1f}s")
    if wiki.quality_score:
        lines.append(f"- **Health score:** {wiki.quality_score:.0f}/100")
    lines.append("")

    # Topic / concept cloud (from edges metadata)
    topics = wiki.metadata.get("topics", [])
    if topics:
        topics_limit = DEFAULTS.get(DISPLAY_WIKI_TOPICS, 30)
        lines.append("## Topics\n")
        lines.append(" ".join(f"`{t}`" for t in sorted(topics)[:topics_limit]))
        lines.append("")

    # Full TOC
    lines.append("## Table of Contents\n")
    for i, node in enumerate(wiki.nodes, 1):
        slug = node.id
        lines.append(f"  {i}. [[{slug}|{node.title}]]")

    return "\n".join(lines)
