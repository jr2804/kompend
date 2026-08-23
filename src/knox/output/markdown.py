"""Markdown output writer — OKF v0.2-compliant vault (default format).

Produces wiki output conformant with the Google Open Knowledge Format v0.2
spec: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md

OKF is a **compliance spec**, not a separate format — this markdown writer is
the only markdown writer, and it is OKF-compliant by default.

Conformance (§11):
1. Every concept ``.md`` file has parseable YAML frontmatter.
2. Every frontmatter has a non-empty ``type`` field.
3. Reserved filenames (``index.md``, ``log.md``) follow §8/§9 structure when present.

What this writer emits:
- Frontmatter is the **canonical OKF v0.2 schema** built once by
  :meth:`VaultWriter.okf_frontmatter` (shared with every vault writer). It
  emits ``type``, ``title``, ``description``, ``tags``, ``status``,
  ``generated: {by, at}``, optional ``verified``, and structured ``sources``
  objects.
- Cross-links use bundle-relative paths: ``[Title](concepts/title.md)`` —
  portable Markdown that also renders in Obsidian (§6.1 recommends absolute
  links; relative is chosen for portability).
- Per-claim attribution uses markdown footnotes keyed to ``sources[].id``
  (§5.1): inline ``[sources/x.pdf]`` citations are rewritten to ``[^id]``
  references and the definitions are emitted at the foot of each page.
- ``index.md`` has grouped sections with descriptions (§8). Root index declares
  ``okf_version: "0.2"`` — the only frontmatter permitted in an index (§12).
- ``log.md`` follows OKF §9: date-grouped ``## YYYY-MM-DD`` headings,
  newest first; entries are ``- [HH:mm:ss] **Action**: prose`` list items
  that accumulate across recompiles, ingests, queries, and lint passes
  (never wiped).
- ``references.md`` is a ``type: Reference`` concept with a formatted bibliography.

Two modes:
- **Vault mode** (default): one ``.md`` per node. Concept pages are organized
  under ``concepts/``; the overview and index stay at root.
- **Single-file mode**: a single ``wiki.md`` containing all nodes as sections.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, ClassVar, override

from knox.compilation.concept_keys import nearest_alias, normalize_concept_key
from knox.core.models import CompiledWiki, MetadataField, Reference, WikiNode
from knox.core.paths import DEFAULT_OUTPUT_DIR_NAME, INDEX_FILE, LOG_FILE, QUALITY_FILE, SUMMARIES_SUBDIR
from knox.core.pipeline.assets import AssetStore
from knox.core.pipeline.references import ReferenceStore
from knox.core.utils.time import iso_local
from knox.output.models import Document, OkfSource
from knox.output.pages.log import append_log_entry
from knox.output.renderer import DocumentRenderer
from knox.output.vault import OKF_VERSION, VaultWriter
from knox.output.vault import okf_frontmatter_yaml as _okf_frontmatter

# Regex to find [[wikilinks]] or [[wikilinks|alias]] in body text.
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")

# Match in-text asset references: ``¶ <id>``.
_ASSET_FOOTNOTE_RE = re.compile(r"¶ ([\w-]+)")

# Match inline source citations ``[sources/<file>.pdf]`` or multi-citations
# ``[sources/a.pdf, sources/b.pdf]``.
_CITATION_RE = re.compile(r"\[sources/[^\]\n]+\]")

# Match inline footnote references ``[^id]``.
_FOOTNOTE_REF_RE = re.compile(r"\[\^([^\]]+)\]")


# _NoAliasDumper + okf_frontmatter_yaml live in vault.py (shared with the
# per-source summary writer); markdown imports them below.


class MarkdownWriter(VaultWriter):
    """OKF v0.2-compliant vault writer (markdown, the default format).

    Implements the flavor-specific rendering hooks on top of
    :class:`VaultWriter`'s shared vault algorithm (stale-clearing, link
    resolution, citation rewriting, page orchestration). Frontmatter is the
    canonical OKF v0.2 schema from :meth:`VaultWriter.okf_frontmatter`.
    """

    name: ClassVar[str] = "markdown"

    def __init__(
        self,
        flavor: str | None = None,
        frontmatter_defaults: dict[str, Any] | None = None,
        rumdl_config: Path | None = None,
        vault_mode: bool = True,
        max_related_links: int = 15,
        references_file: Path | None = None,
        assets_file: Path | None = None,
        base_dir: Path | None = None,
        output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
    ) -> None:
        super().__init__(
            max_related_links=max_related_links,
            references_file=references_file,
            assets_file=assets_file,
            base_dir=base_dir,
            output_dir_name=output_dir_name,
            flavor=flavor,
        )
        self.frontmatter_defaults = frontmatter_defaults or {}
        self.renderer = DocumentRenderer(flavor=flavor, rumdl_config=rumdl_config)
        self.vault_mode = vault_mode

    def write(self, wiki: CompiledWiki, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = self._wiki_to_documents(wiki)

        if not self.vault_mode:
            markdown = self.renderer.render_all(docs, title=wiki.metadata.get("title", "Compiled Wiki"))
            output_file = output_dir / "wiki.md"
            output_file.write_text(markdown, encoding="utf-8")
            return output_file

        return self._write_vault(docs, wiki, output_dir)

    def _load_ref_store(self) -> ReferenceStore | None:
        """Load the reference store once if the references file exists."""
        if self._references_file and self._references_file.exists():
            return ReferenceStore(self._references_file)
        return None

    @override
    def _wiki_to_documents(self, wiki: CompiledWiki) -> list[Document]:
        """Build Documents with canonical OKF v0.2 frontmatter.

        Structured ``sources`` are resolved from the :class:`ReferenceStore`;
        ``verified`` trust events are read from ``wiki.metadata`` when present.
        """
        docs: list[Document] = []
        compiled_at = iso_local()
        ref_store = self._load_ref_store()
        verified = wiki.metadata.get("verified")
        verification = self._load_verification()

        for node in wiki.nodes:
            sources = _provenance_sources(node, ref_store)
            frontmatter = VaultWriter.okf_frontmatter(
                node,
                compiled_at,
                self.frontmatter_defaults,
                sources=sources or None,
                verified=verified,
            )
            # Per-page verification outcome (machine-readable, no body noise).
            if verification is not None:
                quality = VaultWriter.node_quality(verification, node.id)
                if quality:
                    frontmatter["verified_claims"] = f"{quality['supported']}/{quality['claims']}"
                    frontmatter["quality_score"] = quality["quality_score"]

            body = node.content or ""
            images = node.metadata.get("images", [])
            if images:
                folder = self._folder_for_kind(node.node_type)
                images = self.portablize_images(
                    images,
                    node.metadata.get(MetadataField.PROVENANCE) or [],
                    folder,
                )
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
        slug_targets: dict[str, str],
    ) -> str:
        """Render one page: frontmatter + body (relative links) + Related + footnotes."""
        rendered = _okf_frontmatter(doc.frontmatter)
        prefix = "#" * min(doc.heading_level, 6)
        rendered += f"{prefix} {doc.title}\n\n"

        # Link paths must be relative to THIS page's directory (a page inside
        # concepts/ links to ``x.md``, not ``concepts/x.md``).
        folder = self._folder_for_kind(doc.node_type)
        page_targets = {k: _relativize(v, folder) for k, v in slug_targets.items()} if folder else slug_targets

        # Convert wikilinks to portable relative markdown links, then rewrite
        # inline source citations to footnote references (§5.1).
        body = convert_wikilinks(body, page_targets)
        sources = doc.frontmatter.get("sources") or []
        if sources:
            body = rewrite_citations_to_footnotes(body, sources)
        rendered += body
        if body and not body.endswith("\n"):
            rendered += "\n"

        # Related section with standard markdown links (not wikilinks).
        if related:
            rendered += "\n## Related\n\n"
            for title, rel_type, confidence in related[: self.max_related_links]:
                path = page_targets.get(title, "")
                link = f"[{title}]({path})" if path else title
                annotation = f" ({rel_type})" if rel_type else ""
                if confidence is not None:
                    annotation += f" — confidence {confidence:.2f}"
                rendered += f"- {link}{annotation}\n"

        if sources:
            footnotes = _format_source_footnotes(rendered, sources)
            if footnotes:
                rendered += "\n" + footnotes

        # Asset footnotes: ¶ <id> references in body → [^<id>]: <summary>.
        asset_store: AssetStore | None = None
        if self._assets_file and self._assets_file.exists():
            asset_store = AssetStore(self._assets_file)
        if asset_store:
            asset_footnotes = _format_asset_footnotes(rendered, asset_store)
            if asset_footnotes:
                rendered += "\n" + asset_footnotes

        return rendered

    @override
    def _write_index(self, docs: list[Document], output_dir: Path, wiki: CompiledWiki) -> None:
        """Write ``index.md`` per §8: grouped sections with descriptions.

        Root index MAY declare ``okf_version`` in frontmatter (§12 exception).
        Group headings are ``##`` so the bundle keeps a single top-level H1.
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
            lines.append(f"## {type_name}\n")
            for doc in sorted(group, key=lambda d: d.title):
                folder = self._folder_for_kind(doc.node_type)
                relative = f"{folder}/{doc.page_slug}.md" if folder else f"{doc.page_slug}.md"
                description = doc.frontmatter.get("description", "")
                entry = f"- [{doc.title}]({relative})"
                lines.append(f"{entry} - {description}" if description else entry)
            lines.append("")

        # Generated surfaces beyond concept pages: summaries + references +
        # quality. Complete-catalog invariant: everything the wiki generates
        # is reachable from this index.
        summaries = sorted((output_dir / SUMMARIES_SUBDIR).glob("*.md")) if (output_dir / SUMMARIES_SUBDIR).is_dir() else []
        if summaries:
            lines.append("## Sources\n")
            lines.extend(f"- [{p.stem}](summaries/{p.name})" for p in summaries)
            lines.append("")
        surfaces: list[tuple[str, str]] = [("references.md", "Corpus sources + cited works not in corpus")]
        if (output_dir / QUALITY_FILE).exists():
            surfaces.append((QUALITY_FILE, "Claim verification summary (WiCER pass)"))
        if (output_dir / "overview.md").exists():
            surfaces.append(("overview.md", "Thematic overview of the corpus"))
        if (output_dir / LOG_FILE).exists():
            surfaces.append((LOG_FILE, "Append-only activity journal"))
        surfaces.append(("../attachments/", "Media & figures extracted from sources"))
        if surfaces:
            lines.append("## References & surfaces\n")
            for name, desc in surfaces:
                lines.append(f"- [{name}]({name}) — {desc}")
            lines.append("")

        (output_dir / INDEX_FILE).write_text("\n".join(lines), encoding="utf-8")

    @override
    def _write_log(self, output_dir: Path, wiki: CompiledWiki) -> None:
        """Write ``log.md`` per OKF §9: ``## YYYY-MM-DD`` headings, newest first.

        Entries survive recompiles — the log is appended to, never wiped.
        """
        article_count = len(wiki.nodes)
        source_count = wiki.metadata.get(MetadataField.SOURCE_COUNT, 0)
        append_log_entry(
            output_dir / LOG_FILE,
            action="Creation",
            detail=f"Compiled {article_count} page(s) from {source_count} source(s).",
        )


def _relativize(bundle_path: str, folder: str) -> str:
    """Rebase a bundle-relative path onto the current page's folder."""
    rel = os.path.relpath(bundle_path, folder)
    return rel.replace("\\", "/")


def convert_wikilinks(body: str, slug_targets: dict[str, str]) -> str:
    """Convert ``[[wikilinks]]`` to portable bundle-relative markdown links.

    ``slug_targets`` maps page slugs/titles to bundle-relative paths
    (e.g. ``"my-page" -> "concepts/my-page.md"``). The lookup is
    alias-aware: short forms like ``[[P.863]]`` resolve to a page whose
    long title contains the alias, mirroring the lint rule's behaviour
    so the rendered output stays consistent with what the wiki considers
    resolvable. Unknown targets are left as Obsidian-style wikilinks
    (``[[target]]``) so the user can see broken cross-links in their
    markdown viewer rather than silently losing them.
    """
    # Build a richer lookup index from slug_targets keys. ``slug_targets``
    # is keyed by slug + long title; we need the bare-acronym and
    # normalized forms as well.
    alias_index: dict[str, str] = dict(slug_targets)
    alias_index.update(_alias_keys(slug_targets.keys(), slug_targets))

    def _replace(match: re.Match[str]) -> str:
        target = match.group(1)
        alias = match.group(2) or target
        # 1. Direct lookup (slug or long title)
        path = alias_index.get(target)
        # 2. Alias-aware nearest-match (short forms like P.863).
        # nearest_alias returns the value at the best-scoring key — which
        # here is the path itself, so we can use it directly.
        if path is None:
            path = nearest_alias(target, alias_index)
        if path is not None:
            return f"[{alias}]({path})"
        # 3. Unknown — leave as Obsidian wikilink so the user sees it.
        return match.group(0)

    return _WIKILINK_RE.sub(_replace, body)


def _alias_keys(titles: Any, paths: dict[str, str]) -> dict[str, str]:
    """Build alias-aware lookup keys for :func:`convert_wikilinks`.

    Mirrors :func:`knox.lint.structural._node_lookup_index` — acronym,
    bare-prefix-before-paren, and slug forms — so the rendered output
    resolves the same targets the lint rule does. Returns a dict of
    alias-key → path suitable for ``setdefault`` merging.
    """
    out: dict[str, str] = {}
    for key in titles:
        path = paths[key]
        stripped = normalize_concept_key(key)
        if stripped and stripped != key.lower():
            out.setdefault(stripped, path)
        before_paren = key.split("(", 1)[0].strip().lower()
        if before_paren and before_paren != key.lower():
            out.setdefault(before_paren, path)
    return out


def rewrite_citations_to_footnotes(body: str, sources: list[OkfSource]) -> str:
    """Rewrite inline ``[sources/<file>.pdf]`` citations to ``[^id]`` refs.

    The footnote id is resolved from the source entry (by its ``id`` or by the
    stem of its ``resource``), so it works whether ids are file stems or
    structured reference ids. Unmatched citations stay as readable plain text
    (no orphan footnote references). Multi-source citations
    ``[sources/a.pdf, sources/b.pdf]`` become ``[^a][^b]``.
    """
    id_by_stem: dict[str, str] = {}
    for src in sources:
        sid = str(src.get("id") or "")
        if not sid:
            continue
        id_by_stem.setdefault(sid.lower(), sid)
        resource = str(src.get("resource") or "")
        if resource:
            id_by_stem.setdefault(Path(resource).stem.lower(), sid)

    def _replace(match: re.Match[str]) -> str:
        refs: list[str] = []
        for raw in match.group(0)[1:-1].split(","):
            item = raw.strip()
            if not item.startswith("sources/"):
                continue
            stem = Path(item[len("sources/") :]).stem
            sid = id_by_stem.get(stem.lower())
            if sid:
                refs.append(f"[^{sid}]")
        return "".join(refs) if refs else match.group(0)

    return _CITATION_RE.sub(_replace, body)


def _provenance_sources(
    node: WikiNode,
    ref_store: ReferenceStore | None,
) -> list[OkfSource]:
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

    sources: list[OkfSource] = []
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


def _reference_to_okf_source(ref: Reference) -> OkfSource:
    """Build an OKF v0.2 ``sources`` entry (§5.1) from a CSL-aligned Reference.

    The ``id`` is the join key for per-claim footnote attribution (§5.1);
    ``resource`` is REQUIRED within an entry and prefers a URL/DOI when present.
    """
    resource = ref.url or (f"https://doi.org/{ref.doi}" if ref.doi else "")
    if not resource:
        # Fall back to the citing source path so the entry always has a resource.
        resource = ref.citing_source or ref.id
    entry: OkfSource = {"id": ref.id or ref.citing_source or ref.short_label()}
    if resource:
        entry["resource"] = resource
    if ref.title:
        entry["title"] = ref.title
    author = ref.display_authors()
    if author:
        entry["author"] = author
    return entry


def _format_source_footnotes(rendered: str, sources: list[OkfSource]) -> str:
    """Render footnote definitions for OKF §5.1 per-claim attribution.

    Only sources whose ``[^id]`` is actually referenced in *rendered* get a
    definition, deduplicated by id and ordered by first reference — no
    orphan definitions (MD066), no orphan references (MD067), no ordering
    warnings.
    """
    ref_positions: dict[str, int] = {}
    for m in _FOOTNOTE_REF_RE.finditer(rendered):
        ref_positions.setdefault(m.group(1), m.start())
    if not ref_positions:
        return ""

    by_id: dict[str, OkfSource] = {}
    for src in sources:
        sid = str(src.get("id") or "")
        if sid and sid not in by_id:
            by_id[sid] = src

    ordered = sorted(
        (s for s in by_id.values() if str(s.get("id")) in ref_positions),
        key=lambda s: ref_positions[str(s["id"])],
    )
    lines: list[str] = []
    for src in ordered:
        sid = str(src["id"])
        label = src.get("title") or src.get("resource") or sid
        lines.append(f"[^{sid}]: {label}")
    return "\n".join(lines) + "\n"


def _format_asset_footnotes(body: str, store: AssetStore) -> str:
    """Render asset footnote definitions for ``¶ <id>`` references in *body*.

    Scans *body* for asset references like ``¶ tab-a1f3`` and ``¶ fig-9c2e``,
    collects the unique asset ids, looks up their summaries from *store*,
    and emits ``[^<id>]: <summary>`` lines OKF-style.
    """
    if not body:
        return ""
    ids_found: set[str] = set()
    for m in _ASSET_FOOTNOTE_RE.finditer(body):
        ids_found.add(m.group(1))
    if not ids_found:
        return ""
    lines: list[str] = []
    for aid in sorted(ids_found):
        record = store.get(aid)
        if record is None:
            continue
        label = record.summary or aid
        lines.append(f"[^{aid}]: {label}")
    return "\n".join(lines) + "\n"
