"""Markdown output writer — uses ``DocumentRenderer`` for flavor support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from knox.core.models import CompiledWiki
from knox.core.paths import INDEX_FILE
from knox.output.base import BaseOutputWriter
from knox.output.models import Document
from knox.output.renderer import DocumentRenderer, render_frontmatter


class MarkdownWriter(BaseOutputWriter):
    """Write compiled wiki as Markdown, optionally formatted to a specific flavor.

    Two output modes:
    - **Vault mode** (default): One ``.md`` file per wiki node with
      ``[[wikilinks]]`` between related pages and an ``index.md`` TOC.
    - **Single-file mode**: Legacy ``wiki.md`` containing all nodes as sections.
    """

    name: ClassVar[str] = "markdown"

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

        for node in wiki.nodes:
            frontmatter = dict(self.frontmatter_defaults)
            frontmatter.update(node.metadata.get("frontmatter", {}))
            frontmatter.setdefault("title", node.title or "Untitled")

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
            )
            docs.append(doc)

        return docs

    # ── Vault mode ──────────────────────────────────────────────────────

    def _write_vault(self, docs: list[Document], wiki: CompiledWiki, output_dir: Path) -> Path:
        """Write each document as a separate ``.md`` file, plus ``index.md``."""
        node_map = {doc.page_slug: doc.title for doc in docs if doc.page_slug}

        related_edges = self._build_related_edges(wiki.edges, node_map)

        for doc in docs:
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

            (output_dir / filename).write_text(rendered, encoding="utf-8")

        self._write_index(docs, output_dir, wiki)
        return output_dir / INDEX_FILE

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
