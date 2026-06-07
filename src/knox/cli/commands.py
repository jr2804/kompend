"""KNOX CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from knox.cli.app import app
from knox.cli.args import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPILE_FORMAT,
    DEFAULT_COMPILE_MODE,
    DEFAULT_IMAGE_MODE,
    DEFAULT_LLM_FALLBACK_MAX_RETRIES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_BUDGET,
    DEFAULT_SOURCES_DIR,
    BypassReview,
    ChunkSize,
    CompileFiles,
    CompileMode,
    EmbedModel,
    ForceTier,
    Frontmatter,
    Ignore,
    ImageMode,
    IncludePrivate,
    IndexStore,
    JsonOutput,
    ListTiers,
    LlmFallbackChain,
    LlmFallbackMaxRetries,
    NoCacheFlag,
    OnlineUrls,
    OutputDir,
    OutputFlavor,
    ParseFiles,
    PrintElements,
    Private,
    Progress,
    PromptBudget,
    Provenance,
    Recursive,
    ReviewMode,
    SourcesDir,
    SynthesizeFlag,
    TargetFormat,
    TwoPhase,
    Verbose,
    Verify,
    Workers,
)
from knox.core.models import CompiledWiki
from knox.core.pipeline import PipelineConfig, build_pipeline
from knox.preprocessing.parsers._registry import register_parser
from knox.preprocessing.parsers.base import ParserRouter
from knox.preprocessing.parsers.docling_parser import DoclingParser
from knox.preprocessing.parsers.pandoc_parser import PandocParser
from knox.preprocessing.parsers.pdfplumber_parser import PdfPlumberParser
from knox.rag.zvec_store import ZvecWikiStore

from ._sources import download_urls, ensure_in_sources, resolve_files
from .doctor import print_doctor_results, run_doctor
from .status import collect_stats, print_status

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
    ignore: Ignore = None,
    show_progress: Progress = None,
    mode: CompileMode = DEFAULT_COMPILE_MODE,
    no_cache: NoCacheFlag = False,
    workers: Workers = 1,
    force: ForceTier = False,
    list_tiers: ListTiers = False,
    two_phase: TwoPhase = True,
    prompt_budget: PromptBudget = DEFAULT_PROMPT_BUDGET,
    review: ReviewMode = False,
    bypass_review: BypassReview = False,
    provenance: Provenance = False,
    private: Private = False,
    verify: Verify = False,
    flavor: OutputFlavor = None,
    frontmatter: Frontmatter = None,
    synthesize: SynthesizeFlag = False,
    llm_fallback_chain: LlmFallbackChain = None,
    llm_fallback_max_retries: LlmFallbackMaxRetries = DEFAULT_LLM_FALLBACK_MAX_RETRIES,
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

    # 1. Resolve file arguments (with optional ignore patterns)
    resolved = resolve_files(
        *files,
        recursive=recursive,
        ignore_patterns=ignore,
    )
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
        all_input_files = resolve_files(
            str(sources_dir_path),
            recursive=recursive,
            ignore_patterns=ignore,
        )
    if not all_input_files:
        typer.echo(f"No supported files found in {sources_dir_path}.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Compiling {len(all_input_files)} document(s) -> {output_dir_path}")

    # 6. Build and run the pipeline
    fallback_chain_list = list(llm_fallback_chain) if llm_fallback_chain else []

    # Parse frontmatter defaults from key=value pairs
    frontmatter_defaults: dict[str, str] = {}
    if frontmatter:
        for kv in frontmatter:
            if "=" in kv:
                k, v = kv.split("=", 1)
                frontmatter_defaults[k.strip()] = v.strip()

    config = PipelineConfig(
        source_dir=sources_dir_path,
        output_dir=output_dir_path,
        online_sources=[],  # already downloaded above
        compile_target=target_format,
        chunk_size=chunk_size,
        image_mode=image_mode,
        input_files=all_input_files,
        ignore_patterns=list(ignore) if ignore else [],
        mode=mode,
        no_cache=no_cache,
        workers=workers,
        force=force,
        flavor=flavor,
        frontmatter_defaults=frontmatter_defaults if frontmatter_defaults else None,
        llm_fallback_chain=fallback_chain_list,
        llm_fallback_max_retries_per_model=llm_fallback_max_retries,
    )

    pipeline = build_pipeline(config, two_phase=two_phase, prompt_budget=prompt_budget)

    if list_tiers:
        _print_tier_listing(pipeline)
        return

    try:
        wiki = pipeline.run()
        typer.echo(f"Compiled wiki written to {output_dir_path}")

        if review and not bypass_review:
            _send_to_review_queue(wiki, output_dir_path)
            typer.echo(f"Pages pending review in {output_dir_path / '.knox' / 'review'}")
        elif bypass_review and review:
            typer.echo("--bypass-review overrides --review; writing pages directly.")

        if provenance:
            _annotate_provenance(wiki, output_dir_path)

        if private:
            _mark_private(wiki, output_dir_path)

        if verify:
            _verify_claims(wiki, output_dir_path)

        # N11: Generate index.md + log.md
        from knox.output.wiki_index import write_index_and_log, write_overview  # noqa: PLC0415

        generated = write_index_and_log(wiki, output_dir_path)
        typer.echo(f"Index: {generated['index']}")
        typer.echo(f"Log:   {generated['log']}")

        if synthesize:
            overview_path = write_overview(wiki, output_dir_path)
            if overview_path:
                typer.echo(f"Overview: {overview_path}")
            else:
                typer.echo("Overview generation skipped (LLM client unavailable).", err=True)

        if index_store:
            _build_vector_index(wiki, output_dir_path, embed_model)

    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


# ── doctor ──────────────────────────────────────────────────────────────


@app.command(name="doctor")
def doctor_cmd(
    verbose: Verbose = False,
    json_output: JsonOutput = False,
) -> None:
    """Validate configuration and connectivity.

    Runs health checks for source/output directories, parser availability,
    environment variables, and optionally LLM connectivity.
    """
    sources_dir = Path(DEFAULT_SOURCES_DIR) if Path(DEFAULT_SOURCES_DIR).exists() else None
    output_dir = Path(DEFAULT_OUTPUT_DIR) if Path(DEFAULT_OUTPUT_DIR).exists() else None
    results = run_doctor(
        sources_dir=sources_dir,
        output_dir=output_dir,
        verbose=verbose,
    )
    exit_code = print_doctor_results(results, json_output=json_output)
    if exit_code:
        raise typer.Exit(1)


# ── status ──────────────────────────────────────────────────────────────


@app.command(name="status")
def status_cmd(
    json_output: JsonOutput = False,
) -> None:
    """Show per-wiki statistics.

    Displays source count, article count, word count, last compile time,
    and config path.
    """
    sources_dir = Path(DEFAULT_SOURCES_DIR) if Path(DEFAULT_SOURCES_DIR).exists() else None
    output_dir = Path(DEFAULT_OUTPUT_DIR) if Path(DEFAULT_OUTPUT_DIR).exists() else None
    stats = collect_stats(sources_dir=sources_dir, output_dir=output_dir)
    print_status(stats, json_output=json_output)


# ── parser subcommand group ─────────────────────────────────────────────


@app.command(name="list-parsers")
def list_parsers_cmd(
    json_output: JsonOutput = False,
) -> None:
    """List all registered parsers (built-in + external + HTTP)."""
    # Ensure built-in parsers are registered, then read from the registry
    _build_router()
    from knox.preprocessing.parsers._registry import get_registered_parsers  # noqa: PLC0415

    parsers = get_registered_parsers()

    if json_output:
        import json as j  # noqa: PLC0415

        j.dump(parsers, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    if not parsers:
        typer.echo("No parsers registered.")
        return

    typer.echo(f"Registered parsers ({len(parsers)}):")
    for p in parsers:
        exts = ", ".join(p.get("extensions", []) or [])
        ptype = p.get("type", "built-in")
        typer.echo(f"  • {p['name']} ({ptype})  [{exts}]")


# ── write (Q18) ─────────────────────────────────────────────────────────


@app.command(name="write")
def write_cmd(
    page_title: str = typer.Argument(..., help="Title of the page to create."),
    source: list[str] = typer.Option([], "--source", "-s", help="Source file(s) to include (repeatable)."),
    frontmatter: Frontmatter = None,
) -> None:
    """Create a one-shot wiki page from a title and optional sources.

    Example:
        knox write "Transformer architecture" --source attention.pdf --source bert.pdf --frontmatter language=en
    """
    # Parse frontmatter from key=value pairs
    fm: dict[str, str] = {}
    if frontmatter:
        for kv in frontmatter:
            if "=" in kv:
                k, v = kv.split("=", 1)
                fm[k.strip()] = v.strip()

    # Collect source content
    body_parts: list[str] = []
    if source:
        from ._sources import resolve_files  # noqa: PLC0415

        source_files = resolve_files(*source)
        for sf in source_files:
            try:
                content = sf.read_text(encoding="utf-8", errors="replace")[:2000]
                body_parts.append(f"From {sf.name}:\n\n{content}\n")
            except Exception as exc:
                typer.echo(f"Warning: could not read {sf}: {exc}", err=True)

    body = "\n\n".join(body_parts) if body_parts else "_(Content pending)_"

    # Render via flvor-aware MarkdownWriter
    from knox.output.markdown import MarkdownWriter  # noqa: PLC0415

    writer = MarkdownWriter()
    from knox.output.models import Document  # noqa: PLC0415

    doc = Document(title=page_title, body=body, frontmatter=fm)
    rendered = writer.renderer.render(doc)

    output_path = Path.cwd() / "wiki" / f"{page_title.lower().replace(' ', '-')}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    typer.echo(f"Written: {output_path}")


# ── provenance (Q16) ────────────────────────────────────────────────────


@app.command(name="provenance")
def provenance_cmd(
    page_slug: str = typer.Argument(..., help="Page slug or title to show provenance for."),
    with_content: bool = typer.Option(False, "--with-content", help="Show cited content snippets."),
    json_output: JsonOutput = False,
) -> None:
    """Show source-to-article provenance for a compiled wiki page.

    Example:
        knox provenance self-attention
        knox provenance self-attention --with-content
    """
    # Read the compiled wiki
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    wiki_file = output_dir / "wiki.md"
    if not wiki_file.exists():
        typer.echo(f"No compiled wiki found at {wiki_file}. Run 'knox compile' first.", err=True)
        raise typer.Exit(1)

    # Try loading the wiki from cache
    wiki: CompiledWiki | None = None
    cache_file = output_dir / ".knox" / "last_wiki.json"
    if cache_file.exists():
        try:
            import json as j  # noqa: PLC0415

            data = j.loads(cache_file.read_text(encoding="utf-8"))
            from knox.core.models import CompiledWiki  # noqa: PLC0415

            wiki = CompiledWiki(**data)
        except Exception:
            pass

    if wiki is None:
        typer.echo("Could not load compiled wiki metadata.", err=True)
        raise typer.Exit(1)

    # Find the matching node
    nodes = [n for n in wiki.nodes if n.id == page_slug or n.title.lower() == page_slug.lower()]
    if not nodes:
        typer.echo(f"Page '{page_slug}' not found in compiled wiki.", err=True)
        raise typer.Exit(1)

    node = nodes[0]
    source_chunks = node.source_chunks or []

    if json_output:
        import json as j  # noqa: PLC0415

        data = {"page": node.title, "slug": node.id, "sources": source_chunks}
        j.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    typer.echo(f"# {node.title}\n")
    if source_chunks:
        for chunk_id in source_chunks:
            typer.echo(f"  Source: {chunk_id}")
    else:
        typer.echo("  (no provenance tracked)")


# ── Internal helpers ────────────────────────────────────────────────────


def _build_router() -> ParserRouter:
    router = ParserRouter()

    try:
        docling = DoclingParser()
        router.register(docling)
        register_parser("DoclingParser", "built-in", list(docling.supported_extensions))
    except Exception:
        pass

    try:
        pdfplumber = PdfPlumberParser()
        router.register(pdfplumber)
        register_parser("PdfPlumberParser", "built-in", list(pdfplumber.supported_extensions))
    except Exception:
        pass

    try:
        pandoc = PandocParser()
        router.register(pandoc)
        register_parser("PandocParser", "built-in", list(pandoc.supported_extensions))
    except Exception:
        pass

    return router


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
        typer.echo("Warning: no nodes indexed — vector index skipped", err=True)
    store.close()


# ── New feature helpers (Q20/N3/N8/N4/Q21) ──────────────────────────────


def _print_tier_listing(pipeline: object) -> None:
    """Print the tier listing from a WikiPipeline."""
    if not hasattr(pipeline, "list_tiers"):
        typer.echo("Pipeline does not support tier listing.", err=True)
        return
    import json as j  # noqa: PLC0415

    typer.echo(j.dumps(pipeline.list_tiers(), indent=2, default=str))


def _send_to_review_queue(wiki: CompiledWiki, output_dir_path: Path) -> None:
    """Send compiled wiki nodes to the review queue."""
    from knox.review import ReviewQueue  # noqa: PLC0415

    queue = ReviewQueue(review_dir=output_dir_path / ".knox" / "review")
    for node in wiki.nodes:
        queue.enqueue(node)


def _annotate_provenance(wiki: CompiledWiki, output_dir_path: Path) -> None:
    """Annotate compiled wiki with claim-level provenance citations."""
    from knox.provenance import annotate_wiki_with_provenance  # noqa: PLC0415

    annotated = annotate_wiki_with_provenance(wiki, output_dir_path)
    typer.echo(f"Annotated {len(annotated.nodes)} nodes with provenance")


def _mark_private(wiki: CompiledWiki, output_dir_path: Path) -> None:
    """Mark all wiki pages as private via frontmatter."""
    from knox.trust import mark_wiki_private  # noqa: PLC0415

    mark_wiki_private(wiki, output_dir_path)


def _verify_claims(wiki: CompiledWiki, output_dir_path: Path) -> None:
    """Run grounding verification over wiki claims."""
    from knox.verification import verify_wiki  # noqa: PLC0415

    result = verify_wiki(wiki, output_dir_path)
    typer.echo(f"Verified {result.verified}/{result.total} claims (avg confidence {result.average_confidence:.2f})")
