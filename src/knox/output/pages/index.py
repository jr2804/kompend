"""Generate ``index.md`` — TOC + stats for the compiled wiki."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from wenmode.nodes import Paragraph, Text

from knox.core.config import DEFAULTS
from knox.core.config.keys import DISPLAY_WIKI_TOPICS
from knox.core.models import CompiledWiki, MetadataField, WikiField
from knox.core.page_builder import PageBuilder, link, strong
from knox.core.paths import EXPORTED_DIR_NAME, INDEX_FILE, LOG_FILE, SUMMARIES_SUBDIR
from knox.core.utils.time import display_local
from knox.output.vault import VaultWriter

EXPORTED_HEADING: Final[str] = "Exported sources"


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
        return _hub_index(wiki, prefix, base_dir)
    return _full_catalog(wiki, output_dir, prefix, base_dir)


def sync_exported_sources_section(index_file: Path, base_dir: Path) -> None:
    """Insert or refresh the ``## Exported sources`` section in *index_file*.

    Idempotent: replaces the section between the heading and the next ``## ``
    heading (or the file end) with a fresh listing from ``exported/``, or
    removes it when the directory is empty. No-op when *index_file* does
    not exist — ``write_index`` renders the section on the next compile.

    The generation-time index writers render the exact same section from
    the same on-disk scan, so the splice and a later recompile agree byte
    for byte (memory #44: single source of truth, no hand-maintained list).
    """
    if not index_file.exists():
        return
    section = exported_sources_section(base_dir)
    text = index_file.read_text(encoding="utf-8")
    updated = _splice_section(text, section)
    if updated != text:
        index_file.write_text(updated, encoding="utf-8")


def _splice_section(text: str, section: str) -> str:
    """Replace/append/remove the exported-sources block in *text*."""
    heading = f"## {EXPORTED_HEADING}"
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln == heading), None)
    if start is None:
        if not section.strip():
            return text
        return "\n\n".join(b for b in (text.strip(), section.strip()) if b) + "\n"
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    prefix = "\n".join(lines[:start]).strip()
    suffix = "\n".join(lines[end:]).strip()
    blocks = [b for b in (prefix, section.strip(), suffix) if b]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _hub_index(wiki: CompiledWiki, prefix: str, base_dir: Path | None) -> str:
    """Thin hub: quick facts + pointer to the portable catalog."""
    b = PageBuilder()
    b.heading(1, "Wiki Index")
    _emit_stat_paragraphs(b, wiki, prefix)
    b.paragraph("The full catalog lives at ", link(f"{prefix}/index.md", f"{prefix}/index.md"), ".")
    section = exported_sources_section(base_dir) if base_dir is not None else ""
    return b.render() + ("\n\n" + section if section else "")


def _full_catalog(
    wiki: CompiledWiki,
    output_dir: Path | None,
    prefix: str,
    base_dir: Path | None = None,
) -> str:
    """Full catalog: stats, topics, per-page TOC, generated surfaces."""
    b = PageBuilder()
    b.heading(1, "Wiki Index")

    b.heading(2, "Summary")
    _emit_stat_paragraphs(b, wiki, prefix)

    # Topic / concept cloud (from edges metadata)
    topics = wiki.metadata.get("topics", [])
    if topics:
        topics_limit = DEFAULTS.get(DISPLAY_WIKI_TOPICS, 30)
        b.heading(2, "Topics")
        b.paragraph(" ".join(f"`{t}`" for t in sorted(topics)[:topics_limit]))

    # Full TOC — catalog entries: link + one-line description.
    b.heading(2, "Table of Contents")
    toc_items: list[Paragraph] = []
    for _i, node in enumerate(wiki.nodes, 1):
        slug = node.id
        folder = VaultWriter._folder_for_kind(
            node.node_type or "",
            str(node.metadata.get(MetadataField.ENTITY_TYPE, "")) if (node.node_type or "") == "entity" else "",
        )
        link_target = f"{prefix}/{folder}/{slug}.md" if folder else f"{prefix}/{slug}.md"
        desc = VaultWriter.okf_description(node)
        entry = Paragraph()
        entry.children.append(link(link_target, node.title))
        if desc:
            entry.children.append(Text(value=f" - {desc}"))
        toc_items.append(entry)
    if toc_items:
        b.numbered_list(toc_items)

    # Generated surfaces — complete-catalog invariant: everything the wiki
    # generates plus durable media is reachable from this index.
    surfaces: list[tuple[str, str]] = []
    if output_dir is not None:
        if (output_dir / SUMMARIES_SUBDIR).is_dir() and any((output_dir / SUMMARIES_SUBDIR).glob("*.md")):
            surfaces.append((f"{prefix}/{SUMMARIES_SUBDIR}/", "Per-source overview pages (assets, citations, backlinks)"))
        if (output_dir / "references.md").exists():
            surfaces.append((f"{prefix}/references.md", "Corpus sources + bibliography"))
        if (output_dir / "overview.md").exists():
            surfaces.append((f"{prefix}/overview.md", "Thematic overview of the corpus"))
        if (output_dir / LOG_FILE).exists():
            surfaces.append((f"{prefix}/{LOG_FILE}", "Append-only activity journal"))
    surfaces.append(("attachments/" if not prefix else f"{prefix}/../attachments/", "Media & figures extracted from sources"))
    b.heading(2, "Wiki surfaces")
    surface_items: list[Paragraph] = []
    for target, desc in surfaces:
        name = target.rstrip("/").rsplit("/", 1)[-1]
        entry = Paragraph()
        entry.children.append(link(target, name))
        entry.children.append(Text(value=f" — {desc}"))
        surface_items.append(entry)
    b.bullet_list(surface_items)

    section = exported_sources_section(base_dir) if base_dir is not None else ""
    return b.render() + ("\n\n" + section if section else "")


def exported_sources_section(base_dir: Path) -> str:
    """Render the ``## Exported sources`` section listing ``exported/*.md``.

    Returns ``""`` when the ``exported/`` directory is missing or holds no
    markdown files — the section only exists when there is something to
    list. Links are relative to *base_dir* (the vault root), matching the
    index's location.
    """
    exported_dir = base_dir / EXPORTED_DIR_NAME
    files = sorted(p.name for p in exported_dir.glob("*.md")) if exported_dir.is_dir() else []
    if not files:
        return ""
    b = PageBuilder()
    b.heading(2, EXPORTED_HEADING)
    b.bullet_list([link(f"exported/{name}", name) for name in files])
    return b.render()


def _emit_stat_paragraphs(b: PageBuilder, wiki: CompiledWiki, prefix: str) -> None:
    """Emit the quick-facts block: one paragraph per stat."""
    source_count = wiki.metadata.get(MetadataField.SOURCE_COUNT, 0)
    duration = wiki.metadata.get(WikiField.DURATION_SECONDS)

    stats: list[tuple[str, str]] = [
        ("Sources", str(source_count)),
        ("Articles", str(len(wiki.nodes))),
        ("Total words", f"{sum(len(n.content.split()) for n in wiki.nodes):,}"),
        ("Last compile", display_local("%Y-%m-%d %H:%M")),
    ]
    if duration is not None:
        stats.append(("Duration", f"{duration:.1f}s"))
    if wiki.quality_score:
        stats.append(("Health score", f"{wiki.quality_score:.0f}/100"))

    att_target = f"{prefix}/../attachments/" if prefix else "attachments/"
    b.paragraph(strong("Sources:"), f" {source_count}")
    b.paragraph(strong("Articles:"), f" {len(wiki.nodes)}")
    b.paragraph(strong("Total words:"), f" {sum(len(n.content.split()) for n in wiki.nodes):,}")
    b.paragraph(strong("Last compile:"), f" {display_local('%Y-%m-%d %H:%M')}")
    if duration is not None:
        b.paragraph(strong("Duration:"), f" {duration:.1f}s")
    if wiki.quality_score:
        b.paragraph(strong("Health score:"), f" {wiki.quality_score:.0f}/100")
    b.paragraph(
        strong("Media:"),
        " ",
        link(att_target, "attachments/"),
        " — figures extracted from sources",
    )


def _bundle_prefix(output_dir: Path | None, base_dir: Path | None) -> str:
    """Relative path from *base_dir* to *output_dir*; ``""`` when flat/unknown."""
    if output_dir is None or base_dir is None or output_dir == base_dir:
        return ""
    try:
        rel = output_dir.relative_to(base_dir)
    except ValueError:
        return ""
    return rel.as_posix()
