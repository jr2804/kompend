"""KNOX CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from knox.chunking.hybrid import HybridChunker
from knox.cli.app import app
from knox.cli.args import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPILE_FORMAT,
    DEFAULT_IMAGE_MODE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCES_DIR,
    ChunkSize,
    CompileFiles,
    EmbedModel,
    ImageMode,
    IndexStore,
    OnlineUrls,
    OutputDir,
    ParseFiles,
    PrintElements,
    Recursive,
    SourcesDir,
    TargetFormat,
)
from knox.compilation.graph import GraphCompiler
from knox.core.models import CompiledWiki
from knox.core.pipeline import PipelineConfig, WikiPipeline
from knox.output.base import BaseOutputWriter
from knox.output.json_writer import JsonWriter
from knox.output.markdown import MarkdownWriter
from knox.output.typst import TypstWriter
from knox.preprocessing.parsers.base import ParserRouter
from knox.preprocessing.parsers.docling_parser import DoclingParser
from knox.preprocessing.parsers.pandoc_parser import PandocParser
from knox.preprocessing.parsers.pdfplumber_parser import PdfPlumberParser
from knox.rag.zvec_store import ZvecWikiStore

from ._sources import download_urls, ensure_in_sources, resolve_files

logger = logging.getLogger(__name__)


# ── parse ───────────────────────────────────────────────────────────────


@app.command(name="parse")
def parse_cmd(
    files: ParseFiles,
    recursive: Recursive = False,
    print_elements: PrintElements = False,
) -> None:
    """Parse documents and show their structure.

    Accepts one or more file paths, glob patterns (e.g. *.pdf), or directories.
    Use --recursive to descend into subdirectories.
    """
    resolved = resolve_files(*files, recursive=recursive)
    if not resolved:
        typer.echo("No supported files found.", err=True)
        raise typer.Exit(1)

    router = _build_router()

    for input_file in resolved:
        try:
            doc = router.parse(input_file)
            typer.echo(f"\n{'─' * 60}")
            typer.echo(f"  {doc.title or input_file.name}")
            typer.echo(f"  Pages: {doc.page_count}  |  Elements: {len(doc.elements)}")
            if print_elements:
                for el in doc.elements[:20]:
                    typer.echo(f"    [{el.type.name:10s}] {str(el.content)[:120]}")
                if len(doc.elements) > 20:
                    typer.echo(f"    ... ({len(doc.elements) - 20} more)")
        except Exception as exc:
            typer.echo(f"Error parsing {input_file}: {exc}", err=True)


# ── compile ─────────────────────────────────────────────────────────────


@app.command(name="compile")
def compile_cmd(
    files: CompileFiles,
    sources_dir: SourcesDir = None,
    output_dir: OutputDir = None,
    online_urls: OnlineUrls = None,
    recursive: Recursive = False,
    target_format: TargetFormat = DEFAULT_COMPILE_FORMAT,
    chunk_size: ChunkSize = DEFAULT_CHUNK_SIZE,
    image_mode: ImageMode = DEFAULT_IMAGE_MODE,
    index_store: IndexStore = False,
    embed_model: EmbedModel = None,
) -> None:
    """Compile documents into a knowledge wiki.

    Source files are gathered from the given paths (files, globs, directories)
    and consolidated into the sources directory. If a file is already there it
    stays in place; otherwise it is copied in. URLs are downloaded into the
    sources directory as well.

    The sources directory and output directory default to 'sources' and 'wiki'
    in the current working directory respectively.
    """
    if online_urls is None:
        online_urls = []
    sources_dir_path = Path(sources_dir) if sources_dir else Path(DEFAULT_SOURCES_DIR)
    output_dir_path = Path(output_dir) if output_dir else Path(DEFAULT_OUTPUT_DIR)

    # 1. Resolve file arguments
    resolved = resolve_files(*files, recursive=recursive)
    if not resolved and not online_urls:
        typer.echo("No files or URLs to compile.", err=True)
        raise typer.Exit(1)

    # 2. Ensure sources dir exists
    sources_dir_path.mkdir(parents=True, exist_ok=True)

    # 3. Copy external files into sources dir
    local_files = ensure_in_sources(resolved, sources_dir_path)

    # 4. Download URLs into sources dir
    downloaded = download_urls(online_urls, sources_dir_path)

    # 5. If the only argument was a directory (or no files resolved),
    #    scan the sources dir for all supported files
    all_input_files = list(dict.fromkeys(local_files + downloaded))  # dedupe, preserve order
    if not all_input_files:
        all_input_files = resolve_files(str(sources_dir_path), recursive=recursive)
    if not all_input_files:
        typer.echo(f"No supported files found in {sources_dir_path}.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Compiling {len(all_input_files)} document(s) -> {output_dir_path}")

    # 6. Build and run the pipeline
    config = PipelineConfig(
        source_dir=sources_dir_path,
        output_dir=output_dir_path,
        online_sources=[],  # already downloaded above
        compile_target=target_format,
        chunk_size=chunk_size,
        image_mode=image_mode,
        input_files=all_input_files,
    )

    writer = _get_writer(target_format)

    pipeline = WikiPipeline(
        config=config,
        parser=DoclingParser(image_mode=image_mode),
        chunker=HybridChunker(chunk_size=chunk_size),
        compiler=GraphCompiler(),
        output_writer=writer,
    )

    try:
        wiki = pipeline.run()
        typer.echo(f"Compiled wiki written to {output_dir_path}")

        if index_store:
            _build_vector_index(wiki, output_dir_path, embed_model)

    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


# ── Internal helpers ────────────────────────────────────────────────────


def _build_router() -> ParserRouter:
    router = ParserRouter()
    router.register(DoclingParser())
    router.register(PdfPlumberParser())
    router.register(PandocParser())
    return router


def _get_writer(target_format: str) -> BaseOutputWriter:
    format_map: dict[str, type[BaseOutputWriter]] = {
        "markdown": MarkdownWriter,
        "json": JsonWriter,
        "typst": TypstWriter,
    }
    writer_cls = format_map.get(target_format)
    if writer_cls is None:
        typer.echo(
            f"Error: Unknown format '{target_format}'. Use 'markdown', 'json', or 'typst'.",
            err=True,
        )
        raise typer.Exit(1)
    return writer_cls()


def _build_vector_index(
    wiki: CompiledWiki,
    output_dir_path: Path,
    embed_model: str | None,
) -> None:
    store = ZvecWikiStore(db_path=output_dir_path / ".knox" / "vectors")
    count = store.build_index(wiki, embedding_model=embed_model)
    if count > 0:
        typer.echo(f"Indexed {count} nodes in zvec store for semantic search")
    else:
        typer.echo("Warning: zvec not available — vector index skipped", err=True)
    store.close()
