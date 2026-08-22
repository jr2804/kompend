"""Generate ``index.md`` — TOC + stats for the compiled wiki."""

from __future__ import annotations

from pathlib import Path

from knox.core.config import DEFAULTS
from knox.core.config.keys import DISPLAY_WIKI_TOPICS
from knox.core.models import CompiledWiki, MetadataField, WikiField
from knox.core.paths import INDEX_FILE, LOG_FILE, QUALITY_FILE, SUMMARIES_SUBDIR
from knox.core.utils.time import display_local
from knox.output.vault import VaultWriter


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
    index_content = generate_index(wiki, output_dir=output_dir, base_dir=resolved_base)
    index_file.write_text(index_content, encoding="utf-8")

    return {"index": index_file}


def generate_index(
    wiki: CompiledWiki,
    *,
    output_dir: Path | None = None,
    base_dir: Path | None = None,
) -> str:
    """Generate ``index.md`` — TOC + stats for the compiled wiki.

    TOC entries use portable relative links with one-line descriptions
    (catalog style per the LLM-wiki design); no Obsidian-only syntax.
    """
    prefix = _bundle_prefix(output_dir, base_dir)
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
    lines.append(f"- **Bundle:** [{prefix}/index.md]({prefix}/index.md)")
    lines.append("")

    # Topic / concept cloud (from edges metadata)
    topics = wiki.metadata.get("topics", [])
    if topics:
        topics_limit = DEFAULTS.get(DISPLAY_WIKI_TOPICS, 30)
        lines.append("## Topics\n")
        lines.append(" ".join(f"`{t}`" for t in sorted(topics)[:topics_limit]))
        lines.append("")

    # Full TOC — catalog entries: link + one-line description.
    lines.append("## Table of Contents\n")
    for i, node in enumerate(wiki.nodes, 1):
        slug = node.id
        folder = VaultWriter._KIND_FOLDERS.get(node.node_type or "", "")
        link = f"{prefix}/{folder}/{slug}.md" if folder else f"{prefix}/{slug}.md"
        desc = VaultWriter.okf_description(node)
        entry = f"{i}. [{node.title}]({link})"
        lines.append(f"{entry} - {desc}" if desc else entry)
    lines.append("")

    # Generated surfaces — complete-catalog invariant: everything the wiki
    # generates is reachable from this index.
    surfaces: list[tuple[str, str]] = []
    if output_dir is not None:
        if (output_dir / SUMMARIES_SUBDIR).is_dir() and any((output_dir / SUMMARIES_SUBDIR).glob("*.md")):
            surfaces.append((f"{prefix}/{SUMMARIES_SUBDIR}/", "Per-source overview pages (assets, citations, backlinks)"))
        if (output_dir / "references.md").exists():
            surfaces.append((f"{prefix}/references.md", "Bibliography — all extracted citations, anchored by id"))
        if (output_dir / QUALITY_FILE).exists():
            surfaces.append((f"{prefix}/{QUALITY_FILE}", "Claim verification summary (WiCER pass)"))
        if (output_dir / "overview.md").exists():
            surfaces.append((f"{prefix}/overview.md", "Thematic overview of the corpus"))
        if (output_dir / LOG_FILE).exists():
            surfaces.append((f"{prefix}/{LOG_FILE}", "Append-only activity journal"))
    if surfaces:
        lines.append("## Wiki surfaces\n")
        for target, desc in surfaces:
            name = target.rsplit("/", 1)[-1] or target
            lines.append(f"- [{name}]({target}) — {desc}")

    return "\n".join(lines)


def _bundle_prefix(output_dir: Path | None, base_dir: Path | None) -> str:
    """Portable path prefix from the index location to the bundle pages."""
    if output_dir is not None and base_dir is not None and output_dir.is_relative_to(base_dir):
        return output_dir.relative_to(base_dir).as_posix()
    return output_dir.name if output_dir is not None else "wiki"
