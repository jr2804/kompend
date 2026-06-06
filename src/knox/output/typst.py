"""Typst output writer."""

from __future__ import annotations

from pathlib import Path

from knox.core.models import CompiledWiki
from knox.output.base import BaseOutputWriter


class TypstWriter(BaseOutputWriter):
    """Write compiled wiki as Typst source for PDF generation."""

    def write(self, wiki: CompiledWiki, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "compiled_wiki.typ"
        lines: list[str] = []
        lines.append('#set page("a4")')
        lines.append("#set text(size: 11pt)")
        lines.append("")
        lines.append("= Compiled Wiki")
        lines.append("")
        for node in wiki.nodes:
            lines.append(f"== {node.title}")
            lines.append(node.content)
            lines.append("")
        output_path.write_text("\n".join(lines))
