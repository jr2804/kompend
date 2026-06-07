"""Markdown output writer."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar
from knox.core.models import CompiledWiki, WikiNode
from knox.output.base import BaseOutputWriter


class MarkdownWriter(BaseOutputWriter):
    """Write compiled wiki as a single markdown file."""

    name: ClassVar[str] = "markdown"

    def write(self, wiki: CompiledWiki, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        lines: list[str] = ["# Compiled Wiki\n"]

        # Write TOC
        if wiki.toc:
            lines.append("## Table of Contents\n")
            for entry in wiki.toc:
                title = entry.get("title", "")
                entry_id = entry.get("id", "")
                lines.append(f"- [{title}](#{entry_id})")
            lines.append("")

        # Write nodes as sections with proper markdown headings
        for node in wiki.nodes:
            level = node.metadata.get("heading_level", 1)
            prefix = "#" * min(level + 1, 6)
            lines.append(f"{prefix} {node.title}\n")
            lines.append(node.content + "\n")

            if node.source_chunks:
                lines.append(f"*Source: {len(node.source_chunks)} chunk(s)*\n")

        output_path = output_dir / "wiki.md"
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path