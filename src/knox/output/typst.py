"""Typst output writer — produces Typst source for PDF compilation."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from knox.core.models import CompiledWiki
from knox.output.base import BaseOutputWriter


class TypstWriter(BaseOutputWriter):
    """Write compiled wiki as Typst source for PDF generation."""

    name: ClassVar[str] = "typst"

    def write(self, wiki: CompiledWiki, output_dir: Path) -> Path:  # noqa: PLR6301 — abc override, needs self
        output_dir.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []

        # Document setup
        lines.append('#set page("a4")')
        lines.append("#set text(size: 11pt)")
        lines.append("")

        # Title page
        title = wiki.metadata.get("title", "Compiled Wiki")
        lines.append("#align(center + top)")
        lines.append(f'#text(size: 24pt, weight: "bold")[{title}]')
        lines.append("#v(2cm)")

        if wiki.metadata.get("description"):  # nocheck: wiki container metadata key
            lines.append("#align(center)")
            lines.append(f"#text(size: 12pt)[{wiki.metadata['description']}]")  # nocheck: wiki container metadata key
            lines.append("#v(1cm)")

        lines.append("#pagebreak()")
        lines.append("")

        # Table of contents
        if wiki.toc:
            lines.append("= Table of Contents")
            lines.append("#outline()")
            lines.append("#pagebreak()")
            lines.append("")

            # Also write explicit TOC entries as cross-references
            for _i, entry in enumerate(wiki.toc):
                title = entry.get("title", "")
                entry_id = entry.get("id", "")
                if entry_id:
                    lines.append(f"- #link(<{entry_id}>)[{title}]")
                else:
                    lines.append(f"- {title}")
            lines.append("")
            lines.append("#pagebreak()")
            lines.append("")

        # Nodes as sections with proper heading levels and cross-references
        for node in wiki.nodes:
            level = node.metadata.get("heading_level", 1)
            heading = "=" * min(level + 1, 6)

            node_id = node.id
            lines.append(f"#label(<{node_id}>)")
            lines.append(f"{heading} {node.title}")
            lines.append("")

            # Content
            lines.append(node.content)
            lines.append("")

            # Source chunks reference
            if node.source_chunks:
                lines.append(f"*Source: {len(node.source_chunks)} chunk(s)*")
                lines.append("")

            # Cross-reference back to TOC
            lines.append("#v(0.5cm)")
            lines.append("#text(size: 9pt, fill: gray)[")
            lines.append("  #link(<toc>)[Back to Table of Contents]")
            lines.append("]")
            lines.append("")
            lines.append("#pagebreak()")
            lines.append("")

        output_file = output_dir / "wiki.typ"
        output_file.write_text("\n".join(lines), encoding="utf-8")
        return output_file
