"""Vault-writing algorithm for kompend output writers.

The vault-wide algorithm — one ``.md`` file per wiki node, stale-file clearing,
related-edge resolution, in-text citation rewriting, and the
references/index/log orchestration — is implemented exactly once here in
:class:`VaultWriter`.

OKF (Open Knowledge Format v0.2) frontmatter is the **unique, canonical
standard** every vault writer emits. The single markdown writer
(:class:`~kompend.output.markdown.MarkdownWriter`) is OKF-compliant and uses
portable relative links — no Obsidian-only syntax.

The pieces that are not the vault algorithm are mixed in, one concern each:
:class:`~kompend.output.page.OkfPageMixin` (how one page is written),
:class:`~kompend.output.page_paths.PageLayoutMixin` (which folder a page lands
in), :class:`~kompend.output.surfaces.GeneratedSurfaceMixin` (per-source
summaries, ``references.md``, ``quality.md``).

Subclasses fill in the rendering hooks that differ by format:

- :meth:`_wiki_to_documents` — build the ``Document`` list (frontmatter, etc.)
- :meth:`_render_document` — render one page (body already citation-rewritten)
- :meth:`_write_index` — root index page
- :meth:`_write_log` — bundle log page
"""

from __future__ import annotations

import shutil
from abc import abstractmethod
from pathlib import Path
from typing import Any

from kompend.core.config._configs import DisplayConfig, OutputConfig
from kompend.core.models import CompiledWiki, GraphEdge, ImageRef, MetadataField, image_source_key
from kompend.core.paths import (
    ATTACHMENTS_DIR_NAME,
    DEFAULT_OUTPUT_DIR_NAME,
    DEFAULT_SOURCES_DIR_NAME,
    INDEX_FILE,
    KOMP_DIR,
    LOG_FILE,
    attachment_slug,
)
from kompend.core.pipeline.references import ReferenceStore
from kompend.logging import get_logger
from kompend.output.base import BaseOutputWriter
from kompend.output.frontmatter import OKF_VERSION
from kompend.output.models import Document
from kompend.output.page import OkfPageMixin
from kompend.output.page_paths import PageLayoutMixin, bundle_path, page_stem
from kompend.output.reference_links import rewrite_source_references
from kompend.output.surfaces import GeneratedSurfaceMixin

# OKF spec version every vault writer targets.
__all__ = ["OKF_VERSION"]

logger = get_logger(__name__)


class VaultWriter(OkfPageMixin, PageLayoutMixin, GeneratedSurfaceMixin, BaseOutputWriter):
    """Base class for writers that emit a vault of ``.md`` files.

    Subclasses fill in the rendering hooks that differ between flavors; the
    shared vault algorithm lives here so it is written exactly once. The
    frontmatter is already uniform — see :meth:`okf_frontmatter`.
    """

    def __init__(
        self,
        *,
        max_related_links: int | None = None,
        description_max_chars: int | None = None,
        references_file: Path | None = None,
        assets_file: Path | None = None,
        base_dir: Path | None = None,
        output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
        **kwargs: Any,
    ) -> None:
        """Create a vault writer.

        ``max_related_links`` and ``description_max_chars`` default to the
        config values (``output.max_related_links``,
        ``display.description_max_chars``); callers that hold a resolved
        config pass them explicitly.
        """
        super().__init__(**kwargs)
        self.max_related_links = max_related_links if max_related_links is not None else OutputConfig().max_related_links
        self.description_max_chars = description_max_chars if description_max_chars is not None else DisplayConfig().description_max_chars
        self._references_file = references_file
        self._assets_file = assets_file
        self._base_dir = base_dir
        self._output_dir_name = output_dir_name

    def write(self, wiki: CompiledWiki, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        docs = self._wiki_to_documents(wiki)
        return self._write_vault(docs, wiki, output_dir)

    def portablize_images(
        self,
        images: list[ImageRef],
        provenance: list[str],
        folder: str,
    ) -> list[ImageRef]:
        """Copy ``.kompend/``-internal images into ``attachments/`` and rewrite paths.

        Markdown-only invariant: wiki pages must never reference ``.kompend/``
        (the cache is derived state and may be absent). Any embedded image
        living under ``.kompend/`` is copied to
        ``attachments/<source-slug>/<filename>`` and the reference rewritten to
        a link relative to the page's *folder*. Images outside ``.kompend/`` and
        data URIs pass through unchanged. Requires ``base_dir``; without it
        nothing is rewritten.

        The slug comes from the image's own ``source_file`` matched against the
        page provenance — never from the provenance string itself. Provenance
        is agent-reported and frequently names something that is not a source
        (a venue, a product): slugging it made a directory per invented label
        and filled each with the same copied figures. Images whose source
        matches no provenance entry are dropped, so an unattributable figure
        cannot leave a ``.kompend/`` link behind (the lint rule that enforces the
        invariant forbids one).
        """
        if self._base_dir is None or not images:
            return images
        # Provenance entries arrive as bare stems ("report") or source-relative
        # paths ("sources/report.pdf"); attachment_slug() strips the sources/
        # prefix so both forms land on the same directory.
        prefix = f"{DEFAULT_SOURCES_DIR_NAME}/"
        prov_by_key: dict[str, str] = {}
        for entry in provenance:
            if not entry:
                continue
            bare = entry[len(prefix) :] if entry.startswith(prefix) else entry
            prov_by_key.setdefault(image_source_key(entry), bare)
        out: list[ImageRef] = []
        for img in images:
            raw = img.get("path", "")
            rel = self._kompend_relative(raw)
            if rel is None:
                out.append(img)
                continue
            src_file = self._base_dir / rel
            if not src_file.is_file():
                out.append(img)
                continue
            matched = prov_by_key.get(image_source_key(str(img.get(MetadataField.SOURCE_FILE, ""))))
            slug = attachment_slug(matched) if matched else ""
            if not slug:
                # No provenance entry names this image's source — drop it rather
                # than invent a directory or emit a .kompend/ link.
                logger.debug("Dropping image with no provenance match: %s", rel)
                continue
            dest = self._base_dir / ATTACHMENTS_DIR_NAME / slug / src_file.name
            if not dest.exists() or dest.read_bytes() != src_file.read_bytes():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest)
            target = Path(self._output_dir_name) / folder if folder else Path(self._output_dir_name)
            link = dest.relative_to(self._base_dir).relative_to(target, walk_up=True).as_posix()
            new_img = img.copy()
            new_img["path"] = link
            out.append(new_img)
        return out

    def _write_vault(self, docs: list[Document], wiki: CompiledWiki, output_dir: Path) -> Path:
        """Write each document as a separate ``.md`` file, plus index/log pages."""
        # Clear stale compile output so the on-disk vault exactly reflects the
        # current CompiledWiki. Root ``*.md`` plus generated subfolders are
        # removed; user-authored subfolders (``_distilled/``, ``write/``) stay.
        # ``log.md`` is append-only history (Karpathy) and survives recompiles.
        for stale in output_dir.glob("*.md"):
            if stale.name != LOG_FILE:
                stale.unlink()
        for folder in set(self._KIND_FOLDERS.values()):
            sub = output_dir / folder
            if not sub.is_dir():
                continue
            # rglob, not glob: entity pages live in typed subfolders
            # (``entities/<type>/<slug>.md``), so a non-recursive sweep left
            # every stale page behind whenever the entity-type set changed —
            # producing orphaned pages with broken concept links, plus
            # duplicates when a type was renamed (``event/`` → ``conference/``).
            for stale in sub.rglob("*.md"):
                stale.unlink()
            # Prune typed subfolders left empty by the sweep so the vault does
            # not accumulate directories for entity types that no longer exist.
            # Deepest first, so nested empties collapse in one pass.
            for child in sorted((p for p in sub.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                if next(child.iterdir(), None) is None:
                    child.rmdir()

        # slug/title -> bundle-relative path mapping for link rendering.
        # Paths are relative (no leading slash) so links stay portable.
        slug_targets: dict[str, str] = {}
        for doc in docs:
            folder = self._folder_for_kind(doc.node_type, doc.entity_subtype)
            path = bundle_path(folder, doc.page_slug)
            slug_targets[doc.page_slug] = path
            slug_targets[doc.title] = path

        node_map = {doc.page_slug: doc.title for doc in docs if doc.page_slug}

        related_edges = self._build_related_edges(wiki.edges, node_map)

        # Load reference store once for citation link rewriting.
        ref_store: ReferenceStore | None = None
        if self._references_file is not None:
            ref_store = ReferenceStore(self._references_file)

        for doc in docs:
            folder = self._folder_for_kind(doc.node_type, doc.entity_subtype)
            target_dir = output_dir / folder if folder else output_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{page_stem(doc.page_slug)}.md"

            # Rewrite in-text citation markers to reference links (shared).
            body = doc.body
            if ref_store is not None:
                # Determine the source file from provenance metadata.
                source_files = doc.frontmatter.get("source_files") or []
                if source_files:
                    # Try each source stem against the store's keys.
                    for source_key in self._candidate_source_keys(source_files, ref_store):
                        rewritten = rewrite_source_references(body, source_key, ref_store)
                        if rewritten != body:
                            body = rewritten
                            break

            rendered = self._render_document(doc, body, related_edges.get(doc.page_slug, []), wiki, slug_targets)
            (target_dir / filename).write_text(rendered, encoding="utf-8")

        # References page is always generated — an empty bibliography is
        # information too (signals reference extraction found nothing).
        self._write_references_page(ref_store, output_dir, wiki)
        # Per-source summary pages (markdown-only invariant: per-source asset
        # inventory + citation mapping live in the wiki, not only in .kompend/).
        self._write_source_summaries(wiki, output_dir)
        # Verification summary (WiCER) — per-page claim table from
        # .kompend/verification.json; stays internal when no pass ran.
        self._write_quality_page(wiki, output_dir)

        self._write_index(docs, output_dir, wiki)
        self._write_log(output_dir, wiki)
        return output_dir / INDEX_FILE

    # --- Format-specific hooks (overridden by subclasses) -----------------

    @abstractmethod
    def _wiki_to_documents(self, wiki: CompiledWiki) -> list[Document]: ...

    @abstractmethod
    def _render_document(
        self,
        doc: Document,
        body: str,
        related: list[tuple[str, str, float | None]],
        wiki: CompiledWiki,
        slug_targets: dict[str, str],
    ) -> str: ...

    @abstractmethod
    def _write_index(self, docs: list[Document], output_dir: Path, wiki: CompiledWiki) -> None: ...

    @abstractmethod
    def _write_log(self, output_dir: Path, wiki: CompiledWiki) -> None:
        """Write the bundle log page (no-op for the wikilink variant)."""

    def _kompend_relative(self, raw: str) -> Path | None:
        """Return *raw* as a base_dir-relative path if it lives under ``.kompend/``."""
        if not raw or raw.startswith("data:") or self._base_dir is None:
            return None
        p = Path(raw)
        try:
            rel = p.relative_to(self._base_dir) if p.is_absolute() else p
        except ValueError:
            return None
        return rel if rel.parts and rel.parts[0] == KOMP_DIR else None

    @staticmethod
    def _build_related_edges(
        edges: list[GraphEdge],
        node_map: dict[str, str],
    ) -> dict[str, list[tuple[str, str, float | None]]]:
        """Build a map of page slug -> sorted list of related page entries.

        Each entry is ``(target_title, relation_type, confidence)``. Typed
        relation edges carry their type and confidence; plain entity-reference
        edges are rendered without an annotation.
        """
        related: dict[str, dict[str, tuple[str, float | None]]] = {}
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            rel_type = edge.get("type", "")
            confidence = edge.get("confidence")
            confidence = float(confidence) if isinstance(confidence, (int, float)) else None

            for src, tgt in ((source, target), (target, source)):
                if not src or not tgt or src == tgt:
                    continue
                if src not in node_map or tgt not in node_map:
                    continue
                target_title = node_map[tgt]
                current = related.setdefault(src, {})
                # Prefer the highest-confidence annotation per target.
                prev = current.get(target_title)
                if prev is None or (confidence is not None and (prev[1] is None or confidence > prev[1])):
                    current[target_title] = (rel_type, confidence)

        result: dict[str, list[tuple[str, str, float | None]]] = {}
        for slug, targets in related.items():
            sorted_targets = sorted(
                targets.items(),
                key=lambda item: (item[1][1] if item[1][1] is not None else -1.0, item[0]),
                reverse=True,
            )
            result[slug] = [(title, rel_type, confidence) for title, (rel_type, confidence) in sorted_targets]
        return result

    @staticmethod
    def _candidate_source_keys(sources: list[str], store: ReferenceStore) -> list[str]:
        """Yield source keys likely to match store entries for the given sources.

        The store is keyed by full source paths (``sources/paper.pdf``); the
        frontmatter ``sources`` list contains stems (``paper``). We try
        multiple candidate formats against the store's actual keys.
        """
        store_keys = set(store.to_dict().keys())
        candidates: list[str] = []
        for src in sources:
            for candidate in (
                src,
                f"sources/{src}",
                f"sources/{src}.pdf",
                f"sources/{src}.md",
                f"sources/{src}.txt",
            ):
                if candidate in store_keys:
                    candidates.append(candidate)
            # Also check for partial matches (stem anywhere in key)
            for key in store_keys:
                if src and src in Path(key).stem:
                    candidates.append(key)
        return candidates

    @staticmethod
    def _embed_images(body: str, images: list[ImageRef]) -> str:
        """Embed image references into the document body.

        Each image dict should have:
        - ``path``: file path or base64 data URI
        - ``caption``: optional VLM-generated caption (used as alt text)
        - ``element_type``: "image" (for future extensibility)

        Images are appended after the body text with proper Markdown syntax.
        """
        lines: list[str] = [body.rstrip(), ""] if body.strip() else []

        for img in images:
            img_file = img.get("path", "")
            caption = img.get("caption", "")

            if not img_file:
                continue

            alt_text = caption or "image"
            lines.append(f"![{alt_text}]({img_file})")
            if caption:
                lines.append(f"*{caption}*")
            lines.append("")

        return "\n".join(lines)
