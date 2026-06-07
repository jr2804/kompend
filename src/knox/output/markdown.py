"""Markdown output writer — uses ``DocumentRenderer`` for flavor support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from knox.core.models import CompiledWiki
from knox.output.base import BaseOutputWriter
from knox.output.models import Document
from knox.output.renderer import DocumentRenderer


class MarkdownWriter(BaseOutputWriter):
    """Write compiled wiki as Markdown, optionally formatted to a specific flavor.

    Flavor support (obsidian, gfm, mkdocs, etc.) is provided by
    :class:`DocumentRenderer` which pipes output through ``rumdl fmt``
    when available.
    """

    name: ClassVar[str] = "markdown"

    def __init__(
        self,
        flavor: str | None = None,
        frontmatter_defaults: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(flavor=flavor)
        self.frontmatter_defaults = frontmatter_defaults or {}
        self.renderer = DocumentRenderer(flavor=flavor)

    def write(self, wiki: CompiledWiki, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert CompiledWiki → list[Document]
        docs = self._wiki_to_documents(wiki)

        # Render all documents via DocumentRenderer
        markdown = self.renderer.render_all(docs, title=wiki.metadata.get("title", "Compiled Wiki"))

        output_path = output_dir / "wiki.md"
        output_path.write_text(markdown, encoding="utf-8")
        return output_path

    def _wiki_to_documents(self, wiki: CompiledWiki) -> list[Document]:
        """Convert a :class:`CompiledWiki` to a list of :class:`Document`."""
        docs: list[Document] = []

        for node in wiki.nodes:
            # Merge frontmatter defaults with per-node metadata
            frontmatter = dict(self.frontmatter_defaults)
            frontmatter.update(node.metadata.get("frontmatter", {}))

            doc = Document(
                title=node.title or "Untitled",
                body=node.content or "",
                frontmatter=frontmatter,
                heading_level=node.metadata.get("heading_level", 1),
                source_chunks=node.source_chunks,
                page_slug=node.id,
            )
            docs.append(doc)

        return docs
