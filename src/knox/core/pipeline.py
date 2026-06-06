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
    extract_images: bool = True
    table_format: str = "structured"  # "structured" or "markdown"
    chunk_size: int = 512
    chunk_overlap: int = 128
    compile_target: str = "markdown"  # "markdown", "json", "typst"
    online_sources: list[str] = field(default_factory=list)


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
        resolved = []
        if self.resolver:
            for chunk in all_chunks:
                resolved.append(self.resolver.resolve(chunk))

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
        """Yield source files from the source directory."""
        for ext in ("*.pdf", "*.docx", "*.doc", "*.epub", "*.md", "*.txt", "*.html", "*.pptx"):
            yield from self.config.source_dir.glob(ext)
