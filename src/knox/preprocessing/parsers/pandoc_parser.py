"""Pandoc-based parser for universal document conversion."""

from __future__ import annotations

import subprocess
from pathlib import Path

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.preprocessing.parsers.base import BaseParser


class PandocParser(BaseParser):
    """Parser using pandoc for converting DOCX, EPUB, HTML, etc. to markdown."""

    supported_extensions = (".docx", ".doc", ".epub", ".html", ".pptx", ".odt", ".rst")

    def parse(self, source_path: Path) -> ParsedDocument:
        try:
            result = subprocess.run(
                ["pandoc", str(source_path), "-t", "markdown", "--wrap=none"],
                capture_output=True,
                text=True,
                check=True,
            )
            text = result.stdout
        except subprocess.CalledProcessError:
            text = ""
        except FileNotFoundError:
            raise RuntimeError("pandoc not found. Install pandoc to use PandocParser.")

        elements: list[DocumentElement] = []
        for line in text.split("\n"):
            if line.strip():
                elements.append(
                    DocumentElement(
                        type=ElementType.TEXT,
                        content=line.strip(),
                    )
                )

        return ParsedDocument(
            source_path=source_path,
            elements=elements,
            title=source_path.stem,
            metadata={"parser": "pandoc"},
        )
