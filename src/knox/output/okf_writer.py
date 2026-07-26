"""OKF (Open Knowledge Format) v0.2 output writer (default format).

Produces wiki output conformant with the Google Open Knowledge Format v0.2 spec:
https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md

This is the **default** output format. Use ``--format markdown`` or
``compile.output.format = "markdown"`` to select the Obsidian-flavored vault mode.

OKF v0.2 conformance (§11):
1. Every concept ``.md`` file has parseable YAML frontmatter.
2. Every frontmatter has a non-empty ``type`` field.
3. Reserved filenames (``index.md``, ``log.md``) follow §8/§9 structure when present.

What this writer does (vs. ``MarkdownWriter``, the Obsidian-flavored variant):
- Frontmatter is the **canonical OKF v0.2 schema** built once by
  :meth:`VaultWriter.okf_frontmatter` (shared with every vault writer). It
  emits ``type``, ``title``, ``description``, ``tags``, ``status``,
  ``generated: {by, at}`` (replaces v0.1 ``timestamp``), optional ``verified``,
  and structured ``sources`` objects (replaces the v0.1 body ``# Citations`` list).
- Cross-links use absolute bundle-relative paths: ``[Title](/concepts/title.md)``
  instead of ``[[Title]]`` (§6.1 recommends absolute links).
- Per-claim attribution uses markdown footnotes keyed to ``sources[].id``
  (§5.1), with the footnote definitions emitted at the foot of each page.
- ``index.md`` has grouped sections with descriptions (§8). Root index declares
  ``okf_version: "0.2"`` — the only frontmatter permitted in an index (§12).
- ``log.md`` uses ISO date headings with bold action prefix (§9).
- ``references.md`` is a ``type: Reference`` concept with a formatted bibliography.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, override

import yaml

from knox.core.models import CompiledWiki, MetadataField, WikiNode
from knox.core.paths import INDEX_FILE, LOG_FILE
from knox.core.pipeline.references import ReferenceStore
from knox.core.utils.time import iso_local
from knox.output.markdown import MarkdownWriter
from knox.output.models import Document
from knox.output.reference_links import format_bibliography_entry
from knox.output.vault import OKF_VERSION, VaultWriter

# Regex to find [[wikilinks]] or [[wikilinks|alias]] in body text.
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


# Custom YAML Dumper that never emits anchors/aliases (avoids &id001/*id001
# for repeated list values, which is valid YAML but confuses some consumers).
class _NoAliasDumper(yaml.Dumper):
    """YAML Dumper that disables anchor/alias emission."""

    def ignore_aliases(self, data: object) -> bool:  # type: ignore[override]  # noqa: PLR6301
        return True


def _okf_frontmatter(frontmatter: dict[str, Any]) -> str:
    """Render frontmatter as YAML without anchors/aliases."""
    yaml_str = yaml.dump(
        frontmatter,
        Dumper=_NoAliasDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=True,
    ).strip()
    return f"---\n{yaml_str}\n---\n\n"


def convert_wikilinks(body: str, slug_to_path: dict[str, str]) -> str:
    """Convert ``[[wikilinks]]`` to OKF absolute markdown links.

    ``slug_to_path`` maps page slugs/titles to bundle-relative paths
    (e.g. ``"my-page" -> "/concepts/my-page.md"``).
    Unknown targets are left as plain text (OKF §6.1: broken links are valid).
    """

    def _replace(match: re.Match[str]) -> str:
        target = match.group(1)
        alias = match.group(2) or target
        path = slug_to_path.get(target)
        if path is not None:
            return f"[{alias}]({path})"
        return alias

    return _WIKILINK_RE.sub(_replace, body)


def _reference_to_okf_source(ref: Any) -> dict[str, Any]:
    """Build an OKF v0.2 ``sources`` entry (§5.1) from a CSL-aligned Reference.

    The ``id`` is the join key for per-claim footnote attribution (§5.1);
    ``resource`` is REQUIRED within an entry and prefers a URL/DOI when present.
    """
    resource = ref.url or (f"https://doi.org/{ref.doi}" if ref.doi else "")
    if not resource:
        # Fall back to the citing source path so the entry always has a resource.
        resource = ref.citing_source or ref.id
    entry: dict[str, Any] = {"id": ref.id or ref.citing_source or ref.short_label()}
    if resource:
        entry["resource"] = resource
    if ref.title:
        entry["title"] = ref.title
    author = ref.display_authors()
    if author:
        entry["author"] = author
    return entry


def _provenance_sources(
    node: WikiNode,
    ref_store: ReferenceStore | None,
) -> list[dict[str, Any]]:
    """Build the OKF v0.2 ``sources`` list for a node from its provenance.

    Provenance is a list of source identifiers (filenames/URLs). When a
    :class:`ReferenceStore` is available, each provenance source is expanded
    into one structured ``sources`` entry per extracted reference (§5.1). When
    no structured references exist, the raw provenance identifiers are wrapped
    as minimal ``{id, resource}`` entries so the field stays spec-shaped.
    """
    provenance = node.metadata.get(MetadataField.PROVENANCE) or []
    if not provenance:
        return []

    # Deduplicate provenance identifiers, preserving order.
    seen: set[str] = set()
    prov_ids: list[str] = []
    for p in provenance:
        ps = str(p)
        if ps and ps not in seen:
            seen.add(ps)
            prov_ids.append(ps)

    sources: list[dict[str, Any]] = []
    if ref_store is not None:
        for prov in prov_ids:
            refs = ref_store.get_source(prov) or ref_store.get_source(Path(prov).stem)
            for ref in refs:
                sources.append(_reference_to_okf_source(ref))

    # If no structured references resolved, emit minimal spec-shaped entries.
    if not sources:
        for prov in prov_ids:
            sources.append({"id": Path(prov).stem, "resource": prov})
    return sources


def _format_source_footnotes(sources: list[dict[str, Any]]) -> str:
    """Render footnote definitions for OKF §5.1 per-claim attribution.

    Each ``sources[].id`` becomes a ``[^id]: <title or resource>`` line. The
    body cites a claim with ``[^id]``; the definitions live at the page foot.
    """
    if not sources:
        return ""
    lines: list[str] = []
    for src in sources:
        sid = str(src.get("id") or "")
        if not sid:
            continue
        label = src.get("title") or src.get("resource") or sid
        lines.append(f"[^{sid}]: {label}")
    return "\n".join(lines) + "\n"


class OkfWriter(MarkdownWriter):
    """Write compiled wiki as an OKF v0.2 knowledge bundle.

    Implements only the flavor-specific rendering hooks on top of
    :class:`VaultWriter`'s shared vault algorithm (stale-clearing, link
    resolution, citation rewriting, page orchestration). Frontmatter is the
    canonical OKF v0.2 schema from :meth:`VaultWriter.okf_frontmatter`.
    """

    name: ClassVar[str] = "okf"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(vault_mode=True, **kwargs)

    def _load_ref_store(self) -> ReferenceStore | None:
        """Load the reference store once if the references file exists."""
        if self._references_file and self._references_file.exists():
            return ReferenceStore(self._references_file)
        return None

    @override
    def _wiki_to_documents(self, wiki: CompiledWiki) -> list[Document]:
        """Build Documents with canonical OKF v0.2 frontmatter.

        Delegates to :meth:`VaultWriter.okf_frontmatter` (single source of
        truth for frontmatter). Structured ``sources`` are resolved from the
        :class:`ReferenceStore`; ``verified`` trust events are read from
        ``wiki.metadata`` when present.
        """
        docs: list[Document] = []
        compiled_at = iso_local()
        ref_store = self._load_ref_store()
        verified = wiki.metadata.get("verified")

        for node in wiki.nodes:
            sources = _provenance_sources(node, ref_store)
            frontmatter = VaultWriter.okf_frontmatter(
                node,
                compiled_at,
                self.frontmatter_defaults,
                sources=sources or None,
                verified=verified,
            )

            body = node.content or ""
            images = node.metadata.get("images", [])
            if images:
                body = self._embed_images(body, images)

            docs.append(
                Document(
                    title=node.title or "Untitled",
                    body=body,
                    frontmatter=frontmatter,
                    heading_level=node.metadata.get("heading_level", 1),
                    source_chunks=node.source_chunks,
                    page_slug=node.id,
                    node_type=node.node_type,
                ),
            )

        return docs

    @override
    def _render_document(
        self,
        doc: Document,
        body: str,
        related: list[tuple[str, str, float | None]],
        wiki: CompiledWiki,
        slug_to_path: dict[str, str],
    ) -> str:
        """Render one OKF page: frontmatter + body (absolute links) + Related + footnotes."""
        rendered = _okf_frontmatter(doc.frontmatter)
        prefix = "#" * min(doc.heading_level, 6)
        rendered += f"{prefix} {doc.title}\n\n"

        # Convert wikilinks to OKF absolute bundle-relative links (§6.1).
        body = convert_wikilinks(body, slug_to_path)
        rendered += body
        if body and not body.endswith("\n"):
            rendered += "\n"

        # Related section with standard markdown links (not wikilinks).
        if related:
            rendered += "\n## Related\n\n"
            for title, rel_type, confidence in related[: self.max_related_links]:
                path = slug_to_path.get(title, "")
                link = f"[{title}]({path})" if path else title
                annotation = f" ({rel_type})" if rel_type else ""
                if confidence is not None:
                    annotation += f" — confidence {confidence:.2f}"
                rendered += f"- {link}{annotation}\n"

        # OKF §5.1 per-claim attribution footnotes, keyed to sources[].id.
        sources = doc.frontmatter.get("sources") or []
        if sources:
            footnotes = _format_source_footnotes(sources)
            if footnotes:
                rendered += "\n" + footnotes

        return rendered

    @override
    def _write_references_page(self, store: ReferenceStore, output_dir: Path) -> None:
        """Write ``references.md`` as a ``type: Reference`` concept (§4.1).

        Includes all references across all sources, sorted alphabetically by
        short label. Each entry carries a numbered prefix for legacy readers.
        """
        all_refs = store.all_references()
        if not all_refs:
            return
        all_refs.sort(key=lambda r: r.short_label().lower())

        frontmatter = {
            "okf_version": OKF_VERSION,
            "type": "Reference",
            "title": "References",
        }
        lines: list[str] = [_okf_frontmatter(frontmatter), "# References\n"]
        for i, ref in enumerate(all_refs, 1):
            lines.append(format_bibliography_entry(ref, number=i))

        (output_dir / "references.md").write_text("\n".join(lines), encoding="utf-8")

    @override
    def _write_index(self, docs: list[Document], output_dir: Path, wiki: CompiledWiki) -> None:
        """Write ``index.md`` per §8: grouped sections with descriptions.

        Root index MAY declare ``okf_version`` in frontmatter (§12 exception).
        """
        by_type: dict[str, list[Document]] = {}
        for doc in docs:
            okf_type = doc.frontmatter.get("type", "Concept")
            by_type.setdefault(okf_type, []).append(doc)

        lines: list[str] = [
            "---",
            f'okf_version: "{OKF_VERSION}"',
            "---",
            "",
            f"# {wiki.metadata.get('title', 'Knowledge Bundle')}\n",
        ]

        for type_name in sorted(by_type):
            group = by_type[type_name]
            lines.append(f"# {type_name}\n")
            for doc in sorted(group, key=lambda d: d.title):
                folder = self._folder_for_kind(doc.node_type)
                rel_path = f"{folder}/{doc.page_slug}.md" if folder else f"{doc.page_slug}.md"
                description = doc.frontmatter.get("description", "")
                entry = f"- [{doc.title}](/{rel_path})"
                lines.append(f"{entry} - {description}" if description else entry)
            lines.append("")

        (output_dir / INDEX_FILE).write_text("\n".join(lines), encoding="utf-8")

    @override
    def _write_log(self, output_dir: Path, wiki: CompiledWiki) -> None:
        """Write ``log.md`` per §9: ISO date headings, bold action prefix."""
        log_path = output_dir / LOG_FILE
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        article_count = len(wiki.nodes)
        source_count = wiki.metadata.get(MetadataField.SOURCE_COUNT, 0)
        entry = f"**Creation**: Compiled {article_count} page(s) from {source_count} source(s).\n"

        existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

        date_heading = f"## {today}"
        if date_heading in existing:
            # Prepend entry under the existing date heading.
            existing = existing.replace(date_heading, f"{date_heading}\n{entry}", 1)
            log_path.write_text(existing, encoding="utf-8")
        else:
            new_section = f"# Bundle Update Log\n\n{date_heading}\n{entry}\n"
            if existing:
                stripped = re.sub(r"^# Bundle Update Log\n+", "", existing)
                log_path.write_text(new_section + stripped, encoding="utf-8")
            else:
                log_path.write_text(new_section, encoding="utf-8")
