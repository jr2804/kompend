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
    """Generate ``index.md`` for the vault root.

    When the compiled bundle lives in a subdirectory of the vault root
    (``prefix`` non-empty), this is a **thin hub**: quick facts plus a
    pointer to the portable catalog at ``<prefix>/index.md`` — not a second
    full TOC duplicating it. In flat layouts (bundle == vault root) the full
    catalog is generated here instead, since it is then the only index.
    """
    prefix = _bundle_prefix(output_dir, base_dir)
    nested = output_dir is not None and base_dir is not None and output_dir != base_dir
    if nested:
        return _hub_index(wiki, prefix)
    return _full_catalog(wiki, output_dir, prefix)


def _hub_index(wiki: CompiledWiki, prefix: str) -> str:
    """Thin vault-root hub pointing into the wiki bundle."""
    source_count = wiki.metadata.get(MetadataField.SOURCE_COUNT, 0)
    lines: list[str] = [
        "# Wiki Index\n",
        "This vault's knowledge bundle lives in the "
        f"[{prefix}/]({prefix}/) directory. Start at the portable catalog:",
        "",
        f"→ **[{prefix}/index.md]({prefix}/index.md)**\n",
        "## Quick facts\n",
        f"- **Sources:** {source_count}",
        f"- **Pages:** {len(wiki.nodes)}",
        f"- **Last compile:** {display_local('%Y-%m-%d %H:%M')}",
        "- **Media:** [attachments/](attachments/) — figures extracted from sources",
        "",
    ]
    return "\n".join(lines)


def _stats_lines(wiki: CompiledWiki, prefix: str) -> list[str]:
    """Summary bullet list for the catalog index."""
    source_count = wiki.metadata.get(MetadataField.SOURCE_COUNT, 0)
    duration = wiki.metadata.get(WikiField.DURATION_SECONDS)
    lines = [
        f"- **Sources:** {source_count}",
        f"- **Articles:** {len(wiki.nodes)}",
        f"- **Total words:** {sum(len(n.content.split()) for n in wiki.nodes):,}",
        f"- **Last compile:** {display_local('%Y-%m-%d %H:%M')}",
    ]
    if duration is not None:
        lines.append(f"- **Duration:** {duration:.1f}s")
    if wiki.quality_score:
        lines.append(f"- **Health score:** {wiki.quality_score:.0f}/100")
    att_target = "attachments/"
    if prefix:
        lines.append(f"- **Bundle:** [{prefix}/index.md]({prefix}/index.md)")
        att_target = f"{prefix}/../attachments/"
    lines.append(f"- **Media:** [attachments/]({att_target}) — figures extracted from sources")
    return lines


def _full_catalog(
    wiki: CompiledWiki,
    output_dir: Path | None,
    prefix: str,
) -> str:
    """Full catalog: stats, topics, per-page TOC, generated surfaces."""
    lines: list[str] = ["# Wiki Index\n"]

    lines.append("## Summary\n")
    lines.extend(_stats_lines(wiki, prefix))
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
    # generates plus durable media is reachable from this index.
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
    surfaces.append(("attachments/" if not prefix else f"{prefix}/../attachments/", "Media & figures extracted from sources"))
    if surfaces:
        lines.append("## Wiki surfaces\n")
        for target, desc in surfaces:
            name = target.rstrip("/").rsplit("/", 1)[-1]
            lines.append(f"- [{name}]({target}) — {desc}")

    return "\n".join(lines)


def _bundle_prefix(output_dir: Path | None, base_dir: Path | None) -> str:
    """Portable path prefix from the index location to the bundle pages."""
    if output_dir is not None and base_dir is not None and output_dir.is_relative_to(base_dir):
        rel = output_dir.relative_to(base_dir).as_posix()
        return rel if rel != "." else ""
    return output_dir.name if output_dir is not None else "wiki"
