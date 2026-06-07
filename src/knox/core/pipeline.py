"""Pipeline orchestrator for the KNOX 5-tier architecture."""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knox.chunking.base import BaseChunker
from knox.chunking.hybrid import HybridChunker
from knox.compilation.base import BaseCompiler
from knox.compilation.graph import GraphCompiler
from knox.core.models import Chunk, CompiledWiki, DocumentElement, ElementType, ParsedDocument
from knox.output.base import BaseOutputWriter
from knox.output.json_writer import JsonWriter
from knox.output.markdown import MarkdownWriter
from knox.output.typst import TypstWriter
from knox.preprocessing.downloaders.base import BaseDownloader
from knox.preprocessing.parsers.base import BaseParser
from knox.preprocessing.parsers.docling_parser import DoclingParser
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
    llm_model: str = "llama3.2"
    llm_provider: str = "ollama"
    vlm_model: str = "phi-3-vision"
    online_sources: list[str] = field(default_factory=list)
    input_files: list[Path] | None = None  # explicit file list; None = auto-discover from source_dir
    cache_dir: Path | None = None  # None = .knox/cache inside output_dir
    no_cache: bool = False  # force re-parse even if cached
    workers: int = 1  # parallel parse+chunk workers


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

        # Tier 1+2: Parse + Chunk (with cache and optional parallelism)
        all_chunks = self._parse_and_chunk()

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

    def _parse_and_chunk(self) -> list[Any]:
        """Parse documents and chunk them, with optional caching and parallelism."""
        source_files = list(self.iter_sources())
        if not source_files:
            return []

        workers = max(1, self.config.workers)
        use_cache = not self.config.no_cache

        def _process_one(source_file: Path) -> list[Any]:
            """Parse a single file (cache-aware) and chunk it."""
            parsed = self._load_cache(source_file) if use_cache else None
            if parsed is None:
                if not self.parser:
                    return []
                try:
                    parsed = self.parser.parse(source_file)
                    parsed.elements  # ensure fully materialized
                except Exception:
                    logger.exception("Failed to parse %s", source_file)
                    return []
                if use_cache:
                    self._save_cache(source_file, parsed)

            if not self.chunker:
                return []
            return self.chunker.chunk(parsed)

        all_chunks: list[Any] = []
        if workers > 1 and len(source_files) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_process_one, sf): sf for sf in source_files}
                for future in as_completed(futures):
                    try:
                        all_chunks.extend(future.result())
                    except Exception:
                        sf = futures[future]
                        logger.exception("Worker failed for %s", sf)
        else:
            for sf in source_files:
                all_chunks.extend(_process_one(sf))

        return all_chunks

    @staticmethod
    def _cache_key(source_file: Path) -> str:
        """SHA-256 of file contents for cache key."""
        digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
        return f"{source_file.stem}_{digest[:12]}"

    def _cache_dir(self) -> Path:
        """Resolve the cache directory."""
        if self.config.cache_dir:
            return self.config.cache_dir
        return self.config.output_dir / ".knox" / "cache"

    def _load_cache(self, source_file: Path) -> ParsedDocument | None:
        """Load a cached ParsedDocument if available."""
        cache_path = self._cache_dir() / f"{self._cache_key(source_file)}.json"
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            elements = [
                DocumentElement(
                    type=ElementType(el["type"]),
                    content=el["content"],
                    page=el.get("page"),
                    reading_order=el.get("reading_order"),
                    heading_level=el.get("heading_level"),
                    metadata=el.get("metadata", {}),
                )
                for el in data.get("elements", [])
            ]
            return ParsedDocument(
                source_path=Path(data["source_path"]),
                elements=elements,
                title=data.get("title", ""),
                author=data.get("author", ""),
                language=data.get("language", "en"),
                page_count=data.get("page_count", 0),
                metadata=data.get("metadata", {}),
            )
        except Exception:
            logger.debug("Cache read failed for %s — ignoring", cache_path)
            return None

    def _save_cache(self, source_file: Path, doc: ParsedDocument) -> None:
        """Write a ParsedDocument to cache."""
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{self._cache_key(source_file)}.json"
        try:
            data = {
                "source_path": str(doc.source_path),
                "title": doc.title,
                "author": doc.author,
                "language": doc.language,
                "page_count": doc.page_count,
                "metadata": doc.metadata,
                "elements": [
                    {
                        "type": str(el.type),
                        "content": str(el.content) if not isinstance(el.content, list) else el.content,
                        "page": el.page,
                        "reading_order": el.reading_order,
                        "heading_level": el.heading_level,
                        "metadata": el.metadata,
                    }
                    for el in doc.elements
                ],
            }
            cache_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception:
            logger.debug("Cache write failed for %s — ignoring", cache_path)

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


def build_pipeline(config: PipelineConfig) -> WikiPipeline:
    """Shared factory used by both CLI and MCP.

    Constructs a WikiPipeline with default components:
    - DoclingParser (lazy-fallback)
    - HybridChunker
    - GraphCompiler
    - Output writer selected by ``config.compile_target``
    """
    format_map: dict[str, type[BaseOutputWriter]] = {
        "markdown": MarkdownWriter,
        "json": JsonWriter,
        "typst": TypstWriter,
    }
    writer_cls = format_map.get(config.compile_target, MarkdownWriter)

    parser = None
    try:
        parser = DoclingParser(image_mode=config.image_mode)
    except Exception as exc:
        logger.warning("DoclingParser not available: %s", exc)

    return WikiPipeline(
        config=config,
        parser=parser,
        chunker=HybridChunker(chunk_size=config.chunk_size),
        compiler=GraphCompiler(),
        output_writer=writer_cls(),
    )
