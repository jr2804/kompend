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
    ) -> None:
        super().__init__(flavor=flavor)
        self.frontmatter_defaults = frontmatter_defaults or {}
        self.renderer = DocumentRenderer(flavor=flavor, rumdl_config=rumdl_config)
        self.vault_mode = vault_mode

    def write(self, wiki: CompiledWiki, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = self._wiki_to_documents(wiki)

        if not self.vault_mode:
            markdown = self.renderer.render_all(docs, title=wiki.metadata.get("title", "Compiled Wiki"))
            output_path = output_dir / "wiki.md"
            output_path.write_text(markdown, encoding="utf-8")
            return output_path

        return self._write_vault(docs, wiki, output_dir)

    def _wiki_to_documents(self, wiki: CompiledWiki) -> list[Document]:
        """Convert a :class:`CompiledWiki` to a list of :class:`Document`."""
        docs: list[Document] = []

        for node in wiki.nodes:
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

    # ── Vault mode ──────────────────────────────────────────────────────

    def _write_vault(self, docs: list[Document], wiki: CompiledWiki, output_dir: Path) -> Path:
        """Write each document as a separate ``.md`` file, plus ``index.md``."""
        node_map = {doc.page_slug: doc.title for doc in docs if doc.page_slug}

        adj = self._build_adjacency(wiki.edges)

        for doc in docs:
            filename = f"{doc.page_slug or 'untitled'}.md"
            rendered = self.renderer.render(doc)

            linked_ids = adj.get(doc.page_slug, set())
            if linked_ids:
                related = [node_map[lnk] for lnk in linked_ids if lnk in node_map]
                if related:
                    rendered += "\n## Related\n\n"
                    for title in sorted(related):
                        rendered += f"- [[{title}]]\n"

            (output_dir / filename).write_text(rendered, encoding="utf-8")

        self._write_index(docs, output_dir, wiki)
        return output_dir / "index.md"

    def _build_adjacency(self, edges: list[dict[str, Any]]) -> dict[str, set[str]]:
        """Build undirected adjacency from edge list."""
        adj: dict[str, set[str]] = {}
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            if source:
                adj.setdefault(source, set()).add(target)
            if target:
                adj.setdefault(target, set()).add(source)
        return adj

    def _write_index(self, docs: list[Document], output_dir: Path, wiki: CompiledWiki) -> None:
        """Generate ``index.md`` with a table of contents."""
        title = wiki.metadata.get("title", "Wiki Index")
        lines: list[str] = [f"# {title}\n"]

        for doc in docs:
            slug = doc.page_slug
            link = f"[[{doc.title}]]" if not slug or slug == doc.title else f"[[{doc.title}|{slug}]]"
            lines.append(f"- {link}\n")

        index_content = "".join(lines)
        (output_dir / "index.md").write_text(index_content, encoding="utf-8")
