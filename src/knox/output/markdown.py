"""Markdown output writer — uses ``DocumentRenderer`` for flavor support."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

from knox.core.models import CompiledWiki
from knox.core.paths import INDEX_FILE
from knox.core.pipeline.references import ReferenceStore
from knox.core.utils.time import iso_local
from knox.output.models import Document
from knox.output.reference_links import format_bibliography_entry
from knox.output.renderer import DocumentRenderer, render_frontmatter
from knox.output.vault import VaultWriter


class MarkdownWriter(VaultWriter):
    """Obsidian-flavored vault writer (a markdown *flavor*, not a separate format).

    Emits the same canonical OKF frontmatter as every vault writer
    (:meth:`VaultWriter.okf_frontmatter`). The only difference from the default
    OKF format (:class:`OkfWriter`) is the link *syntax*: related pages use
    Obsidian ``[[wikilinks]]`` instead of absolute markdown links.

    Two modes:
    - **Vault mode** (default): one ``.md`` per node. Synthesized concept pages
      are organized under ``concepts/``; the overview and index stay at root.
    - **Single-file mode**: a single ``wiki.md`` containing all nodes as sections.
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
    ) -> None:
        super().__init__(max_related_links=max_related_links, references_file=references_file, flavor=flavor)
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

    def _wiki_to_documents(self, wiki: CompiledWiki) -> list[Document]:
        """Convert a :class:`CompiledWiki` to a list of :class:`Document`."""
        docs: list[Document] = []
        compiled_at = iso_local()

        for node in wiki.nodes:
            # Canonical OKF frontmatter — identical to every vault writer.
            frontmatter = self.okf_frontmatter(node, compiled_at, self.frontmatter_defaults)

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

    def _render_document(
        self,
        doc: Document,
        body: str,
        related: list[tuple[str, str, float | None]],
        wiki: CompiledWiki,
        slug_to_path: dict[str, str],
    ) -> str:
        """Render one page: frontmatter + body (citation-rewritten) + Related."""
        rendered_doc = replace(doc, body=body) if body != doc.body else doc
        rendered = self.renderer.render(rendered_doc)
        if related:
            rendered += "\n## Related\n\n"
            for title, rel_type, confidence in related[: self.max_related_links]:
                annotation = f" ({rel_type})" if rel_type else ""
                if confidence is not None:
                    annotation += f" — confidence {confidence:.2f}"
                rendered += f"- [[{title}]]{annotation}\n"
        return rendered

    def _write_references_page(self, store: ReferenceStore, output_dir: Path) -> None:
        """Write ``references.md`` with a formatted bibliography.

        Includes all references across all sources, sorted alphabetically
        by short label. Each entry has an anchor for citation link targets.
        """
        all_refs = store.all_references()
        if not all_refs:
            return
        all_refs.sort(key=lambda r: r.short_label().lower())

        lines: list[str] = [
            "---",
            'title: "References"',
            "type: Reference",
            "---",
            "",
            "# References",
            "",
        ]
        for i, ref in enumerate(all_refs, 1):
            lines.append(format_bibliography_entry(ref, number=i))

        (output_dir / "references.md").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    def _write_index(self, docs: list[Document], output_dir: Path, wiki: CompiledWiki) -> None:
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

    def _write_log(self, output_dir: Path, wiki: CompiledWiki) -> None:
        # The Obsidian wikilink variant emits no bundle log page.
        return None
