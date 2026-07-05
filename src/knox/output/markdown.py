"""Markdown output writer — uses ``DocumentRenderer`` for flavor support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from knox.core.models import CompiledWiki
from knox.core.paths import INDEX_FILE
from knox.core.timeutil import iso_local
from knox.output.base import BaseOutputWriter
from knox.output.models import Document
from knox.output.renderer import DocumentRenderer, render_frontmatter


class MarkdownWriter(BaseOutputWriter):
    """Write compiled wiki as Markdown, optionally formatted to a specific flavor.

    Two output modes:
    - **Vault mode** (default): One ``.md`` file per wiki node with
      ``[[wikilinks]]`` between related pages and an ``index.md`` TOC.
      Synthesized concept pages are organized under ``concepts/``; the
      overview and index stay at the vault root.
    - **Single-file mode**: Legacy ``wiki.md`` containing all nodes as sections.
    """

    name: ClassVar[str] = "markdown"

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
        flavor: str | None = None,
        frontmatter_defaults: dict[str, Any] | None = None,
        rumdl_config: Path | None = None,
        vault_mode: bool = True,
        max_related_links: int = 15,
    ) -> None:
        super().__init__(flavor=flavor)
        self.frontmatter_defaults = frontmatter_defaults or {}
        self.renderer = DocumentRenderer(flavor=flavor, rumdl_config=rumdl_config)
        self.vault_mode = vault_mode
        self.max_related_links = max_related_links

    def write(self, wiki: CompiledWiki, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = self._wiki_to_documents(wiki)

        if not self.vault_mode:
            markdown = self.renderer.render_all(docs, title=wiki.metadata.get("title", "Compiled Wiki"))
            output_file = output_dir / "wiki.md"
            output_file.write_text(markdown, encoding="utf-8")
            return output_file

        return self._write_vault(docs, wiki, output_dir)

    def _wiki_to_documents(self, wiki: CompiledWiki) -> list[Document]:
        """Convert a :class:`CompiledWiki` to a list of :class:`Document`."""
        docs: list[Document] = []
        compiled_at = iso_local()

        for node in wiki.nodes:
            frontmatter = dict(self.frontmatter_defaults)
            frontmatter.update(node.metadata.get("frontmatter", {}))
            frontmatter.setdefault("title", node.title or "Untitled")
            # Consistent schema: kind, compiled_at, sources on every page.
            frontmatter.setdefault("kind", node.node_type or "page")
            frontmatter.setdefault("compiled_at", compiled_at)
            # Derive sources list: prefer explicit provenance, fall back to
            # source_chunks stems (best-effort).
            prov = node.metadata.get("provenance") or []
            if prov:
                frontmatter.setdefault(
                    "sources",
                    sorted({Path(str(p)).stem for p in prov if p}),
                )

            body = node.content or ""
            images = node.metadata.get("images", [])
            if images:
                body = self._embed_images(body, images)

            doc = Document(
                title=node.title or "Untitled",
                body=body,
                frontmatter=frontmatter,
                heading_level=node.metadata.get("heading_level", 1),
                source_chunks=node.source_chunks,
                page_slug=node.id,
                node_type=node.node_type,
            )
            docs.append(doc)

        return docs

    # ── Vault mode ──────────────────────────────────────────────────────

    def _write_vault(self, docs: list[Document], wiki: CompiledWiki, output_dir: Path) -> Path:
        """Write each document as a separate ``.md`` file, plus ``index.md``."""
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
        node_map = {doc.page_slug: doc.title for doc in docs if doc.page_slug}

        related_edges = self._build_related_edges(wiki.edges, node_map)

        for doc in docs:
            folder = self._folder_for_kind(doc.node_type)
            target_dir = output_dir / folder if folder else output_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{doc.page_slug or 'untitled'}.md"
            rendered = self.renderer.render(doc)

            entries = related_edges.get(doc.page_slug, [])
            if entries:
                rendered += "\n## Related\n\n"
                for title, rel_type, confidence in entries[: self.max_related_links]:
                    annotation = f" ({rel_type})" if rel_type else ""
                    if confidence is not None:
                        annotation += f" — confidence {confidence:.2f}"
                    rendered += f"- [[{title}]]{annotation}\n"

            (target_dir / filename).write_text(rendered, encoding="utf-8")

        self._write_index(docs, output_dir, wiki)
        return output_dir / INDEX_FILE

    @classmethod
    def _folder_for_kind(cls, node_type: str) -> str:
        """Return the subfolder for a page kind, or '' to stay at the root."""
        return cls._KIND_FOLDERS.get(node_type, "")

    @staticmethod
    def _embed_images(body: str, images: list[dict[str, Any]]) -> str:
        """Embed image references into the document body.

        Each image dict should have:
        - ``path``: file path or base64 data URI
        - ``caption``: optional VLM-generated caption (used as alt text)
        - ``element_type``: "image" (for future extensibility)

        Images are appended after the body text with proper Markdown syntax.
        """
        lines: list[str] = [body.rstrip(), ""] if body.strip() else []

        for img in images:
            img_path = img.get("path", "")
            caption = img.get("caption", "")

            if not img_path:
                continue

            alt_text = caption or "image"
            lines.append(f"![{alt_text}]({img_path})")
            if caption:
                lines.append(f"*{caption}*")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _build_related_edges(
        edges: list[dict[str, Any]],
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
    def _write_index(docs: list[Document], output_dir: Path, wiki: CompiledWiki) -> None:
        """Generate ``index.md`` with a table of contents."""
        title = wiki.metadata.get("title", "Wiki Index")
        fm = render_frontmatter({"title": title})
        lines: list[str] = [f"# {title}\n"]

        for doc in docs:
            slug = doc.page_slug
            link = f"[[{doc.title}]]" if not slug or slug == doc.title else f"[[{doc.title}|{slug}]]"
            lines.append(f"- {link}\n")

        index_content = fm + "".join(lines)
        (output_dir / INDEX_FILE).write_text(index_content, encoding="utf-8")
