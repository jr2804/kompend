"""Shared vault-writing algorithm for KNOX output writers.

The vault layout — one ``.md`` file per wiki node, directory routing by kind,
stale-file clearing, related-edge resolution, in-text citation rewriting, and
the references/index/log page orchestration — is implemented exactly once
here in :class:`VaultWriter`.

OKF (Open Knowledge Format v0.2) frontmatter is the **unique, canonical
standard** every vault writer emits. Markdown *flavors* (Obsidian
``[[wikilinks]]``, GFM, …) change only the link *syntax* in the page body —
never the frontmatter. There is one frontmatter schema; build it with
:meth:`VaultWriter.okf_frontmatter`.

Subclasses fill in the rendering hooks that differ by flavor:

- :meth:`_wiki_to_documents` — build the ``Document`` list (frontmatter, etc.)
- :meth:`_render_document` — render one page (body already citation-rewritten)
- :meth:`_write_references_page` — bibliography page
- :meth:`_write_index` — root index page
- :meth:`_write_log` — bundle log page
"""

from __future__ import annotations

import re
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from knox import __version__ as _KNOX_VERSION
from knox.core.models import CompiledWiki, GraphEdge, ImageRef, MetadataField, WikiNode
from knox.core.paths import INDEX_FILE
from knox.core.pipeline.references import ReferenceStore
from knox.output.base import BaseOutputWriter
from knox.output.models import Document, OkfSource, TrustEvent
from knox.output.reference_links import rewrite_source_references

# OKF spec version every vault writer targets.
OKF_VERSION = "0.2"

# OKF actor convention (§7) for the producer of generated content.
# ``<producer>/<version>`` for agents and tools.
_GENERATED_BY = f"knox/{_KNOX_VERSION or '0.0.0'}"

# Map KNOX node_type values to OKF ``type`` values. OKF types are free-form
# strings (§4.1); these are descriptive and self-explanatory per the spec.
KIND_TO_TYPE: dict[str, str] = {
    "concept": "Concept",
    "comparison": "Comparison",
    "entity": "Entity",
    "overview": "Overview",
    "procedure": "Procedure",
    "section": "Section",
}
DEFAULT_TYPE = "Concept"

# Regex to find ``[[wikilinks]]`` or ``[[wikilinks|alias]]`` in body text.
WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


class VaultWriter(BaseOutputWriter):
    """Base class for writers that emit a vault of ``.md`` files.

    Subclasses fill in the rendering hooks that differ between flavors; the
    shared vault algorithm lives here so it is written exactly once. The
    frontmatter is already uniform — see :meth:`okf_frontmatter`.
    """

    # node_type -> subfolder under the vault root. Kinds not listed here
    # (overview, procedure, section, unknown) stay at the root so the
    # layout never splits a type the compiler didn't anticipate.
    _KIND_FOLDERS: ClassVar[dict[str, str]] = {
        "concept": "concepts",
        "comparison": "concepts",
        "entity": "concepts",
    }

    def __init__(
        self,
        *,
        max_related_links: int = 15,
        references_file: Path | None = None,
        assets_file: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_related_links = max_related_links
        self._references_file = references_file
        self._assets_file = assets_file

    def write(self, wiki: CompiledWiki, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        docs = self._wiki_to_documents(wiki)
        return self._write_vault(docs, wiki, output_dir)

    # ── Canonical OKF frontmatter ────────────────────────────────────────
    # This is the single, unique frontmatter schema every vault writer
    # emits. Markdown flavors differ only in body link syntax, not here.

    @staticmethod
    def okf_type(node_type: str | None) -> str:
        """Map a KNOX node_type to an OKF type string."""
        if not node_type:
            return DEFAULT_TYPE
        return KIND_TO_TYPE.get(node_type, node_type.title() or DEFAULT_TYPE)

    @staticmethod
    def okf_description(node: WikiNode) -> str:
        """Extract a one-line description from a wiki node.

        Uses the first non-empty line of content, with ``[[wikilinks]]``
        reduced to plain text.
        """
        content = (node.content or "").strip()
        if not content:
            return ""
        first_line = content.split("\n")[0].strip()
        return WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), first_line)

    @staticmethod
    def okf_frontmatter(
        node: WikiNode,
        compiled_at: str,
        frontmatter_defaults: dict[str, Any] | None = None,
        *,
        sources: list[OkfSource] | None = None,
        verified: list[TrustEvent] | None = None,
    ) -> dict[str, Any]:
        """Build the canonical OKF v0.2 frontmatter for a wiki node.

        The result is OKF §11-conformant (parseable YAML, non-empty ``type``)
        and carries KNOX extensions permitted by OKF §4.1. Config
        ``frontmatter_defaults`` and per-node ``metadata['frontmatter']``
        overrides are merged in.

        OKF v0.2 fields emitted:
        - ``type``, ``title``, ``description``, ``tags``, ``status`` (§4.1, §5.4)
        - ``generated: {by, at}`` — supersedes the v0.1 ``timestamp`` (§5.2, §13.1)
        - ``verified`` — trust events, only when ``verified`` is provided (§5.2)
        - ``sources`` — structured provenance objects (§5.1), only when
          ``sources`` is provided; bare filename provenance is NOT emitted
          here under the reserved ``sources`` key (it would collide with the
          v0.2 object shape) — see the ``source_files`` extension below.

        KNOX extensions (OKF §4.1 permits additional keys):
        - ``kind`` — original KNOX node_type
        - ``source_files`` — provenance filename stems (distinct from OKF
          ``sources`` to avoid shape collision; freed ``tags`` from this overload)
        """
        provenance = node.metadata.get(MetadataField.PROVENANCE) or []
        source_files = sorted({Path(str(p)).stem for p in provenance if p}) if provenance else []
        node_tags = node.metadata.get(MetadataField.TAGS) or []

        frontmatter: dict[str, Any] = dict(frontmatter_defaults or {})
        # Per-node frontmatter overrides config defaults.
        frontmatter.update(node.metadata.get("frontmatter", {}))

        # OKF v0.2 consolidation: the node ``frontmatter`` override is a legacy
        # escape hatch and may carry v0.1 bare-string ``sources`` (filename
        # stems) or ``tags`` overloaded with provenance. Drop both so the only
        # ``sources`` that survives is the structured §5.1 list passed in below,
        # and ``tags`` stays topic-only (provenance stems live in ``source_files``).
        frontmatter.pop("sources", None)
        frontmatter.pop("compiled_at", None)
        frontmatter.pop("timestamp", None)

        # OKF-required fields (always present, spec-compliant).
        frontmatter["type"] = frontmatter.get("type") or VaultWriter.okf_type(node.node_type)
        frontmatter.setdefault("title", node.title or "Untitled")
        description = VaultWriter.okf_description(node)
        if description:
            frontmatter.setdefault("description", description)
        # OKF lifecycle (§5.4): default stable; absence ⇒ stable per spec.
        frontmatter.setdefault("status", "stable")
        # OKF tags (§4.1): cross-cutting topic tags from compilation.
        # Source-filename provenance no longer overloads ``tags``; it lives in
        # the ``source_files`` extension to match OKF v0.2 semantics.
        if node_tags:
            merged_tags = sorted(set(frontmatter.get("tags", [])) | {str(t) for t in node_tags})  # nocheck: OKF frontmatter key, not wiki metadata
            frontmatter["tags"] = merged_tags  # nocheck: OKF frontmatter key, not wiki metadata

        # OKF trust (§5.2): ``generated`` records how/when content was produced.
        # Supersedes the v0.1 ``timestamp`` field (§13.1).
        frontmatter["generated"] = {"by": _GENERATED_BY, "at": compiled_at}
        # ``verified`` only when trust events are known (absence ⇒ unverified tier).
        if verified:
            frontmatter["verified"] = verified

        # OKF structured provenance (§5.1): list of objects with at least
        # ``resource``. Only emitted when callers supply real structured sources
        # (from the ReferenceStore); never populated with bare filename strings.
        if sources:
            frontmatter.setdefault("sources", sources)

        # KNOX extensions (OKF §4.1 permits additional keys).
        frontmatter["kind"] = node.metadata.get("frontmatter", {}).get("kind", node.node_type or "page")
        if source_files:
            frontmatter["source_files"] = source_files

        return frontmatter

    def _write_vault(self, docs: list[Document], wiki: CompiledWiki, output_dir: Path) -> Path:
        """Write each document as a separate ``.md`` file, plus index/log pages."""
        # Clear stale compile output so the on-disk vault exactly reflects the
        # current CompiledWiki. Root ``*.md`` plus generated subfolders are
        # removed; user-authored subfolders (``_distilled/``, ``write/``) stay.
        for stale in output_dir.glob("*.md"):
            stale.unlink()
        for folder in set(self._KIND_FOLDERS.values()):
            sub = output_dir / folder
            if sub.is_dir():
                for stale in sub.glob("*.md"):
                    stale.unlink()

        # slug/title -> bundle-relative path mapping for link rendering.
        slug_targets: dict[str, str] = {}
        for doc in docs:
            folder = self._folder_for_kind(doc.node_type)
            path = f"/{folder}/{doc.page_slug or 'untitled'}.md" if folder else f"/{doc.page_slug or 'untitled'}.md"
            slug_targets[doc.page_slug] = path
            slug_targets[doc.title] = path

        node_map = {doc.page_slug: doc.title for doc in docs if doc.page_slug}

        related_edges = self._build_related_edges(wiki.edges, node_map)

        # Load reference store once for citation link rewriting.
        ref_store: ReferenceStore | None = None
        if self._references_file and self._references_file.exists():
            ref_store = ReferenceStore(self._references_file)

        for doc in docs:
            folder = self._folder_for_kind(doc.node_type)
            target_dir = output_dir / folder if folder else output_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{doc.page_slug or 'untitled'}.md"

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

        # Generate references page if store has entries.
        if ref_store is not None and ref_store.total_count() > 0:
            self._write_references_page(ref_store, output_dir)

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
    def _write_references_page(self, store: ReferenceStore, output_dir: Path) -> None: ...

    @abstractmethod
    def _write_index(self, docs: list[Document], output_dir: Path, wiki: CompiledWiki) -> None: ...

    @abstractmethod
    def _write_log(self, output_dir: Path, wiki: CompiledWiki) -> None:
        """Write the bundle log page (no-op for the wikilink variant)."""

    @classmethod
    def _folder_for_kind(cls, node_type: str) -> str:
        """Return the subfolder for a page kind, or '' to stay at the root."""
        return cls._KIND_FOLDERS.get(node_type, "")

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
                if target_title not in current or (confidence is not None and (current[target_title][1] is None or confidence > current[target_title][1])):  # ty: ignore
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
