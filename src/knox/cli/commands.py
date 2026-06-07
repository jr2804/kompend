"""KNOX CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from knox.chunking.hybrid import HybridChunker
from knox.cli.app import app
from knox.compilation.graph import GraphCompiler
from knox.core.pipeline import PipelineConfig, WikiPipeline
from knox.output.json_writer import JsonWriter
from knox.output.markdown import MarkdownWriter
from knox.output.typst import TypstWriter
from knox.preprocessing.downloaders.http_downloader import HTTPDownloader
from knox.preprocessing.parsers.docling_parser import DoclingParser
from knox.preprocessing.parsers.pandoc_parser import PandocParser
from knox.preprocessing.parsers.pdfplumber_parser import PdfPlumberParser
from knox.rag.zvec_store import ZvecWikiStore

typer.echo = lambda *args, **kwargs: print(*args)


@app.command(name="compile")
def compile_wiki(
    sources_dir: str = typer.Argument(help="Directory with source documents"),
    output_dir: str = typer.Argument(help="Output directory for compiled wiki"),
    online_urls: list[str] = typer.Option([], "--url", help="URLs to download before processing"),
    target_format: str = typer.Option("markdown", "--format", "-f", help="Output format: markdown, json, typst"),
    chunk_size: int = typer.Option(512, "--chunk-size", help="Chunk size in tokens"),
    image_mode: str = typer.Option("file", "--image-mode", help="Image handling: file, base64"),
    index_store: bool = typer.Option(False, "--index", help="Index compiled wiki in zvec for semantic search"),
    embed_model: str | None = typer.Option(None, "--embed-model", help="Embedding model for zvec indexing"),
) -> None:
    """Compile documents into a knowledge wiki."""
    config = PipelineConfig(
        source_dir=Path(sources_dir),
        output_dir=Path(output_dir),
        online_sources=online_urls,
        compile_target=target_format,
        chunk_size=chunk_size,
        image_mode=image_mode,
    )

    # Select output writer based on format
    format_map = {
        "markdown": MarkdownWriter(),
        "json": JsonWriter(),
        "typst": TypstWriter(),
    }
    writer = format_map.get(target_format)
    if writer is None:
        typer.echo(f"Error: Unknown format '{target_format}'. Use 'markdown', 'json', or 'typst'.", err=True)
        raise typer.Exit(1)

    pipeline = WikiPipeline(
        config=config,
        downloader=HTTPDownloader(),
        parser=DoclingParser(image_mode=image_mode) if DoclingParser.supported_extensions else None,  # type: ignore[arg-type]
        chunker=HybridChunker(chunk_size=chunk_size),
        compiler=GraphCompiler(),
        output_writer=writer,
    )

    try:
        wiki = pipeline.run()
        typer.echo(f"Compiled wiki written to {output_dir}")
        
        if index_store:
            store = ZvecWikiStore(db_path=Path(output_dir) / ".knox" / "vectors")
            count = store.build_index(wiki, embedding_model=embed_model)
            if count > 0:
                typer.echo(f"Indexed {count} nodes in zvec store for semantic search")
            else:
                typer.echo("Warning: zvec not available — vector index skipped", err=True)
            store.close()
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command(name="parse")
def parse_single(
    file_path: str = typer.Argument(help="Path to document to parse"),
    print_elements: bool = typer.Option(False, "--elements", help="Print extracted elements"),
) -> None:
    """Parse a single document and show structure."""
    from pathlib import Path

    from knox.preprocessing.parsers.base import ParserRouter

    router = ParserRouter()
    router.register(DoclingParser())
    router.register(PdfPlumberParser())
    router.register(PandocParser())

    path = Path(file_path)
    try:
        doc = router.parse(path)
        typer.echo(f"Parsed: {doc.title or path.name}")
        typer.echo(f"  Pages: {doc.page_count}")
        typer.echo(f"  Elements: {len(doc.elements)}")
        if print_elements:
            for el in doc.elements[:20]:
                typer.echo(f"    [{el.type.name}] {str(el.content)[:100]}...")
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
