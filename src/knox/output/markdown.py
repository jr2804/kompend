"""Markdown output writer."""

from __future__ import annotations

from pathlib import Path

from knox.core.models import CompiledWiki
from knox.output.base import BaseOutputWriter


class MarkdownWriter(BaseOutputWriter):
    """Write compiled wiki as a single markdown file."""

    def write(self, wiki: CompiledWiki, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "compiled_wiki.md"
        lines: list[str] = []
        lines.append("# Compiled Wiki\n")
        for node in wiki.nodes:
            lines.append(f"## {node.title}\n")
            lines.append(f"{node.content}\n")
        output_path.write_text("\n".join(lines))
