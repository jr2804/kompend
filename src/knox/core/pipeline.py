"""Pipeline orchestrator for the KNOX 5-tier architecture."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knox.chunking.base import BaseChunker
from knox.compilation.base import BaseCompiler
from knox.core.models import CompiledWiki, ParsedDocument
from knox.output.base import BaseOutputWriter
from knox.preprocessing.downloaders.base import BaseDownloader
from knox.preprocessing.parsers.base import BaseParser
from knox.resolution.base import BaseResolver

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the KNOX pipeline."""

    source_dir: Path
    output_dir: Path
    image_mode: str = "file"  # "file" or "base64"
    chunk_size: int = 512
    compile_target: str = "markdown"  # "markdown", "json", "typst"
    online_sources: list[str] = field(default_factory=list)
    input_files: list[Path] | None = None  # explicit file list; None = auto-discover from source_dir


class WikiPipeline:
    """Orchestrates the 5-tier KNOX pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        downloader: BaseDownloader | None = None,
        parser: BaseParser | None = None,
        chunker: BaseChunker | None = None,
        resolver: BaseResolver | None = None,
        compiler: BaseCompiler | None = None,
        output_writer: BaseOutputWriter | None = None,
    ) -> None:
        self.config = config
        self.downloader = downloader
        self.parser = parser
        self.chunker = chunker
        self.resolver = resolver
        self.compiler = compiler
        self.output_writer = output_writer

    def run(self) -> CompiledWiki:
        """Execute the full pipeline from sources to compiled wiki."""
        logger.info("Starting KNOX pipeline: %s", self.config.source_dir)

        # Tier 0: Download online sources
        if self.downloader and self.config.online_sources:
            for url in self.config.online_sources:
                self.downloader.download(url, self.config.source_dir)

        # Tier 1: Parse documents
        parsed_docs: list[ParsedDocument] = []
        if self.parser:
            for source_file in self.iter_sources():
                try:
                    parsed = self.parser.parse(source_file)
                    parsed_docs.append(parsed)
                except Exception:
                    logger.exception("Failed to parse %s", source_file)

        # Tier 2: Chunk
        all_chunks = []
        if self.chunker:
            for doc in parsed_docs:
                chunks = self.chunker.chunk(doc)
                all_chunks.extend(chunks)

        # Tier 3: Resolve
        if self.resolver:
            resolved = [self.resolver.resolve(chunk) for chunk in all_chunks]
        else:
            resolved = list(all_chunks)

        # Tier 4: Compile
        wiki = CompiledWiki(nodes=[], edges=[], toc=[], metadata={})
        if self.compiler:
            wiki = self.compiler.compile(resolved)

        # Tier 5: Output
        if self.output_writer:
            self.output_writer.write(wiki, self.config.output_dir)

        logger.info("KNOX pipeline complete: %s -> %s", self.config.source_dir, self.config.output_dir)
        return wiki

    def iter_sources(self) -> Any:
        """Yield source files to process.

        If ``config.input_files`` is set, yields those directly.
        Otherwise auto-discovers supported files from ``config.source_dir``.
        """
        if self.config.input_files:
            yield from self.config.input_files
            return
        for ext in ("*.pdf", "*.PDF", "*.docx", "*.DOCX", "*.doc", "*.DOC",
                    "*.epub", "*.md", "*.txt", "*.html", "*.pptx"):
            yield from self.config.source_dir.rglob(ext)
