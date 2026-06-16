"""Parse, compile, and ingest pipeline commands."""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import typer

from knox.cli._progress import PipelineProgress
from knox.cli._shared import build_cli_overrides, load_hooks, resolve_project_paths
from knox.cli._sources import download_urls, ensure_in_sources, resolve_files
from knox.cli.app import app
from knox.cli.args import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPILE_FORMAT,
    DEFAULT_COMPILER,
    DEFAULT_IMAGE_MODE,
    DEFAULT_LLM_FALLBACK_MAX_RETRIES,
    DEFAULT_PROMPT_BUDGET,
    DEFAULT_START_AT_TIER,
    BudgetConcepts,
    BudgetGenSegment,
    BypassReview,
    ChunkSize,
    CompileFiles,
    CompileLint,
    CompilerChoice,
    ConfigPath,
    EmbedBatchSize,
    EmbedDim,
    EmbedModel,
    ForceTier,
    Frontmatter,
    Ignore,
    ImageMode,
    IndexStore,
    KindOption,
    ListTiers,
    LlmDefaultModel,
    LlmFallbackChain,
    LlmFallbackMaxRetries,
    LlmRoleModel,
    LlmTimeout,
    MemoryDecayThreshold,
    MemoryEnabled,
    MemoryHalfLifeDays,
    NoCacheFlag,
    NoPrivacy,
    OnlineUrls,
    OutputFlavor,
    ParseFiles,
    PrintElements,
    Private,
    Progress,
    PromptBudget,
    Provenance,
    Recursive,
    RelationConfidenceThreshold,
    ResolveRelations,
    ReviewMode,
    StartAtTier,
    SynthesizeFlag,
    TargetFormat,
    UrlFile,
    Verify,
    VlmModel,
    Workers,
)
from knox.core.config import resolve_config
from knox.core.models import CompiledWiki
from knox.core.pipeline import PipelineConfig, build_pipeline
from knox.core.pipeline_defs import Tier
from knox.lint.runner import LintRunner, report_to_text
from knox.output.wiki_index import generate_stage_log_entry, prepend_log, write_index_and_log, write_overview
from knox.preprocessing.parsers._registry import build_default_router
from knox.provenance import annotate_wiki_with_provenance
from knox.rag import build_vector_index
from knox.review import ReviewQueue
from knox.search.index import build_chunk_index
from knox.trust import mark_wiki_private
from knox.verification import verify_wiki

logger = logging.getLogger(__name__)


# ── helpers ─────────────────────────────────────────────────────────────


def _make_progress_callback(tracker: PipelineProgress) -> Callable[[str, int, int, str], None]:
    """Build a progress callback that translates pipeline events into Rich updates.

    The overall bar is reset per phase:
    - **parse**: bar tracks files (0 → N)
    - **resolve**: bar resets to track chunks (0 → M)
    - **compile** / **output**: bar becomes indeterminate
    """

    def _cb(phase: str, current: int, total: int, detail: str) -> None:
        if phase == "parse":
            if current == 0:
                tracker.start()
            elif detail == "done":
                tracker.update_progress(total, f"Parsed {total} file(s)")
            else:
                tracker.update_progress(current, f"[{current}/{total}] {detail}")
        elif phase == "index":
            if current == 0:
                tracker.start()
                tracker.message("Downloading sources…")
            else:
                tracker.message("Download complete")
        elif phase == "resolve":
            if current == 0:
                tracker.reset_for_phase(total, f"Resolving chunks ({total} total)…")
            elif detail == "done":
                tracker.update_progress(total, "Resolve complete")
            else:
                tracker.update_progress(current, f"Resolving [{current}/{total}]")
        elif phase == "compile":
            if current == 0:
                tracker.reset_for_phase(None, "Compiling wiki…")
            else:
                tracker.message("Compile complete")
        elif phase == "output":
            if current == 0:
                tracker.reset_for_phase(None, "Writing output…")
            else:
                tracker.message("Output complete")

    return _cb


def _resolve_compile_inputs(
    files: CompileFiles,
    online_urls: list[str],
    url_file: UrlFile,
    sources_dir_path: Path,
    recursive: Recursive,
    ignore: Ignore,
) -> list[Path]:
    """Resolve file arguments and URLs into a deduplicated input file list."""
    if url_file is not None:
        for raw_line in url_file.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#"):
                online_urls.append(stripped)

    resolved = resolve_files(*files, recursive=recursive, ignore_patterns=ignore)

    sources_dir_path.mkdir(parents=True, exist_ok=True)
    local_files = ensure_in_sources(resolved, sources_dir_path)
    downloaded = download_urls(online_urls, sources_dir_path)

    all_input_files = list(dict.fromkeys(local_files + downloaded))
    if not all_input_files:
        all_input_files = resolve_files(
            str(sources_dir_path),
            recursive=recursive,
            ignore_patterns=ignore,
        )
    return all_input_files


# ── post-compile helpers ────────────────────────────────────────────────


def _write_wiki_cache(wiki: CompiledWiki, base_dir: Path) -> None:
    """Serialize the compiled wiki to ``.knox/last_wiki.json``.

    This lets standalone ``knox lint`` load the wiki without requiring
    a fresh compile. The cache is overwritten on every successful compile.
    """
    cache_dir = base_dir / ".knox"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "last_wiki.json"

    data = {
        "nodes": [dataclasses.asdict(n) for n in wiki.nodes],
        "edges": list(wiki.edges),
        "toc": list(wiki.toc),
        "quality_score": wiki.quality_score,
        "iteration": wiki.iteration,
        "metadata": dict(wiki.metadata),
    }
    cache_file.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")


def _print_tier_listing(pipeline: object) -> None:
    """Print the tier listing from a WikiPipeline."""
    if not hasattr(pipeline, "list_tiers"):
        typer.echo("Pipeline does not support tier listing.", err=True)
        return

    typer.echo(json.dumps(pipeline.list_tiers(), indent=2, default=str))


def _send_to_review_queue(wiki: CompiledWiki, base_dir: Path) -> None:
    """Send compiled wiki nodes to the review queue."""
    queue = ReviewQueue(review_dir=base_dir / ".knox" / "review")
    for node in wiki.nodes:
        queue.enqueue(node)


def _annotate_provenance(wiki: CompiledWiki, output_dir_path: Path) -> None:
    """Annotate compiled wiki with claim-level provenance citations."""
    annotated = annotate_wiki_with_provenance(wiki, output_dir_path)
    typer.echo(f"Annotated {len(annotated.nodes)} nodes with provenance")


def _mark_private(wiki: CompiledWiki, output_dir_path: Path) -> None:
    """Mark all wiki pages as private via frontmatter."""
    mark_wiki_private(wiki, output_dir_path)


def _verify_claims(wiki: CompiledWiki, output_dir_path: Path, *, model: str, verify_budget: int = 3000) -> None:
    """Run grounding verification over wiki claims."""
    result = verify_wiki(wiki, output_dir_path, model=model, verify_budget=verify_budget)
    typer.echo(f"Verified {result.verified}/{result.total} claims (avg confidence {result.average_confidence:.2f})")


def _run_post_compile_lint(wiki: CompiledWiki, sources_dir_path: Path, *, json_output: bool = False) -> None:
    """Run lint after compilation and print the report."""
    runner = LintRunner(sources_dir=str(sources_dir_path) if sources_dir_path.exists() else None)
    report = runner.run(wiki)
    typer.echo("\n" + report_to_text(report))


def _cli_build_vector_index(
    wiki: CompiledWiki,
    output_dir_path: Path,
    embed_model: str | None,
    *,
    base_dir: Path | None = None,
) -> None:
    """Thin CLI wrapper — delegates to knox.rag and prints status."""
    if not embed_model:
        typer.echo("Error: --embed-model must be provided when --index is used.", err=True)
        raise typer.Exit(1)

    count = build_vector_index(wiki, output_dir_path, embed_model=embed_model, base_dir=base_dir)
    if count > 0:
        typer.echo(f"Indexed {count} nodes in zvec store for semantic search")
    else:
        typer.echo("Warning: no nodes indexed — vector index skipped", err=True)


def _post_compile_actions(
    *,
    wiki: CompiledWiki,
    pipeline_config: PipelineConfig,
    output_dir_path: Path,
    sources_dir_path: Path,
    review: bool | None,
    bypass_review: bool | None,
    provenance: bool | None,
    private: bool | None,
    verify: bool | None,
    synthesize: bool | None,
    index_store: bool | None,
    compile_lint: bool | None,
    embed_model: str | None,
) -> None:
    """Run all post-compilation steps (review, provenance, index, lint, etc.)."""
    base_dir = pipeline_config.resolved_base_dir()
    _write_wiki_cache(wiki, base_dir)

    if review and not bypass_review:
        _send_to_review_queue(wiki, base_dir)
        typer.echo(f"Pages pending review in {base_dir / '.knox' / 'review'}")
    elif bypass_review and review:
        typer.echo("--bypass-review overrides --review; writing pages directly.")

    if provenance:
        _annotate_provenance(wiki, output_dir_path)

    if private:
        _mark_private(wiki, output_dir_path)

    if verify:
        resolved_model = pipeline_config.llm.default_model
        _verify_claims(wiki, output_dir_path, model=resolved_model, verify_budget=pipeline_config.budget.verify_output_max * 4)

    generated = write_index_and_log(wiki, output_dir_path, base_dir=base_dir)
    typer.echo(f"Index: {generated['index']}")
    typer.echo(f"Log:   {generated['log']}")

    if synthesize:
        resolved_model = pipeline_config.llm.default_model
        overview_path = write_overview(
            wiki,
            base_dir,
            model=resolved_model,
            overview_budget=pipeline_config.budget.overview_output_max * 4,
            timeout=pipeline_config.llm.timeout,
        )
        if overview_path:
            typer.echo(f"Overview: {overview_path}")
        else:
            typer.echo("Overview generation skipped (LLM client unavailable).", err=True)

    if index_store:
        search_db_path = base_dir / ".knox" / "search.db"
        build_chunk_index(wiki, search_db_path)
        wiki.metadata["embedding_model"] = embed_model
        wiki.metadata["embedding_dim"] = pipeline_config.embedding.dim
        _cli_build_vector_index(wiki, output_dir_path, pipeline_config.embedding.model, base_dir=base_dir)

    if compile_lint:
        _run_post_compile_lint(wiki, sources_dir_path, json_output=False)

    _write_wiki_cache(wiki, base_dir)


# ── commands ────────────────────────────────────────────────────────────


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

    router = build_default_router()

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


@app.command(name="compile")
def compile_cmd(
    ctx: typer.Context,
    files: CompileFiles,
    config: ConfigPath = None,
    online_urls: OnlineUrls = None,
    url_file: UrlFile = None,
    recursive: Recursive = False,
    target_format: TargetFormat = DEFAULT_COMPILE_FORMAT,
    chunk_size: ChunkSize = DEFAULT_CHUNK_SIZE,
    image_mode: ImageMode = DEFAULT_IMAGE_MODE,
    index_store: IndexStore = False,
    embed_model: EmbedModel = None,
    ignore: Ignore = None,
    show_progress: Progress = None,
    start_at_tier: StartAtTier = DEFAULT_START_AT_TIER,
    no_cache: NoCacheFlag = False,
    workers: Workers = 1,
    force: ForceTier = False,
    list_tiers: ListTiers = False,
    prompt_budget: PromptBudget = DEFAULT_PROMPT_BUDGET,
    review: ReviewMode = False,
    bypass_review: BypassReview = False,
    provenance: Provenance = False,
    private: Private = False,
    no_privacy: NoPrivacy = False,
    verify: Verify = False,
    flavor: OutputFlavor = None,
    frontmatter: Frontmatter = None,
    kind: KindOption = None,
    compiler: CompilerChoice = DEFAULT_COMPILER,
    synthesize: SynthesizeFlag = False,
    llm_fallback_chain: LlmFallbackChain = None,
    llm_fallback_max_retries: LlmFallbackMaxRetries = DEFAULT_LLM_FALLBACK_MAX_RETRIES,
    llm_default_model: LlmDefaultModel = None,
    vlm_model: VlmModel = None,
    resolve_relations: ResolveRelations = True,
    relation_confidence_threshold: RelationConfidenceThreshold = 0.5,
    memory_enabled: MemoryEnabled = True,
    memory_half_life_days: MemoryHalfLifeDays = 90.0,
    memory_decay_threshold: MemoryDecayThreshold = 0.3,
    llm_role_model: LlmRoleModel = None,
    compile_lint: CompileLint = False,
    budget_gen_segment: BudgetGenSegment = None,
    budget_concepts: BudgetConcepts = None,
    embed_dim: EmbedDim = None,
    embed_batch_size: EmbedBatchSize = None,
    llm_timeout: LlmTimeout = None,
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

    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cli_overrides = build_cli_overrides(
        wiki_dir=wiki_dir_override,
        target_format=target_format,
        chunk_size=chunk_size,
        image_mode=image_mode,
        start_at_tier=start_at_tier,
        no_cache=no_cache,
        workers=workers,
        force=force,
        review=review,
        provenance=provenance,
        private=private,
        no_privacy=no_privacy,
        verify=verify,
        synthesize=synthesize,
        index_store=index_store,
        compile_lint=compile_lint,
        flavor=flavor,
        frontmatter=frontmatter,
        ignore=ignore,
        llm_default_model=llm_default_model,
        vlm_model=vlm_model,
        resolve_relations=resolve_relations,
        relation_confidence_threshold=relation_confidence_threshold,
        memory_enabled=memory_enabled,
        memory_half_life_days=memory_half_life_days,
        memory_decay_threshold=memory_decay_threshold,
        llm_fallback_chain=llm_fallback_chain,
        llm_fallback_max_retries=llm_fallback_max_retries,
        llm_role_model=llm_role_model,
        kind=kind,
        compiler=compiler,
        budget_gen_segment=budget_gen_segment,
        budget_concepts=budget_concepts,
        embed_dim=embed_dim,
        embed_model=embed_model,
        llm_timeout=llm_timeout,
    )
    cfg = resolve_config(cli_overrides=cli_overrides, config_path=config)
    sources_dir_path, output_dir_path, _cache_dir, _state_dir = resolve_project_paths(cfg)

    all_input_files = _resolve_compile_inputs(
        files,
        online_urls,
        url_file,
        sources_dir_path,
        recursive,
        ignore,
    )
    if not all_input_files:
        typer.echo(f"No supported files found in {sources_dir_path}.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Compiling {len(all_input_files)} document(s) -> {output_dir_path}")

    pipeline_config = PipelineConfig.from_resolved(
        cfg,
        source_dir=sources_dir_path,
        output_dir=output_dir_path,
        input_files=all_input_files,
    )

    pipeline = build_pipeline(pipeline_config)
    load_hooks()

    if list_tiers:
        _print_tier_listing(pipeline)
        return

    tracker = PipelineProgress.create(
        total_files=len(all_input_files),
        force_progress=show_progress,
    )
    progress_cb = _make_progress_callback(tracker) if tracker.enabled else None

    try:
        start_time = time.monotonic()
        wiki = pipeline.run(progress_callback=progress_cb)
        elapsed = time.monotonic() - start_time

        wiki.metadata["duration_seconds"] = round(elapsed, 2)
        wiki.metadata["source_count"] = len(all_input_files)
        wiki.metadata["source_names"] = [Path(f).name for f in all_input_files[:20]]

        tracker.finish()
        typer.echo(f"Compiled wiki written to {output_dir_path}")

        _post_compile_actions(
            wiki=wiki,
            pipeline_config=pipeline_config,
            output_dir_path=output_dir_path,
            sources_dir_path=sources_dir_path,
            review=review,
            bypass_review=bypass_review,
            provenance=provenance,
            private=private,
            verify=verify,
            synthesize=synthesize,
            index_store=index_store,
            compile_lint=compile_lint,
            embed_model=embed_model,
        )

    except Exception as exc:
        tracker.stop()
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command(name="ingest")
def ingest_cmd(
    ctx: typer.Context,
    files: CompileFiles,
    config: ConfigPath = None,
    online_urls: OnlineUrls = None,
    url_file: UrlFile = None,
    recursive: Recursive = False,
    chunk_size: ChunkSize = DEFAULT_CHUNK_SIZE,
    image_mode: ImageMode = DEFAULT_IMAGE_MODE,
    ignore: Ignore = None,
    show_progress: Progress = None,
    no_cache: NoCacheFlag = False,
    workers: Workers = 1,
    force: ForceTier = False,
    llm_default_model: LlmDefaultModel = None,
    vlm_model: VlmModel = None,
) -> None:
    """Stage and parse documents into chunks (Tier 0 + Tier 1).

    Downloads URLs and copies local files into the sources directory,
    then parses and chunks them. Results are cached for subsequent
    'knox compile' runs.

    After ingest, run 'knox compile' to resolve, compile, and render
    the wiki.

    Examples:
        knox ingest *.pdf
        knox ingest --url-file urls.txt
    """
    if online_urls is None:
        online_urls = []

    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cli_overrides = build_cli_overrides(
        wiki_dir=wiki_dir_override,
        chunk_size=chunk_size,
        image_mode=image_mode,
        no_cache=no_cache,
        workers=workers,
        force=force,
        ignore=ignore,
        llm_default_model=llm_default_model,
        vlm_model=vlm_model,
    )
    cfg = resolve_config(cli_overrides=cli_overrides, config_path=config)
    sources_dir_path, output_dir_path, _cache_dir, _state_dir = resolve_project_paths(cfg)

    all_input_files = _resolve_compile_inputs(
        files,
        online_urls,
        url_file,
        sources_dir_path,
        recursive,
        ignore,
    )
    if not all_input_files:
        typer.echo(f"No supported files found in {sources_dir_path}.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Ingesting {len(all_input_files)} document(s) -> {sources_dir_path}")

    pipeline_config = PipelineConfig.from_resolved(
        cfg,
        source_dir=sources_dir_path,
        output_dir=output_dir_path,
        input_files=all_input_files,
    )
    pipeline_config.stop_at_tier = Tier.CHUNK

    pipeline = build_pipeline(pipeline_config)
    load_hooks()

    tracker = PipelineProgress.create(
        total_files=len(all_input_files),
        force_progress=show_progress,
    )
    progress_cb = _make_progress_callback(tracker) if tracker.enabled else None

    try:
        ingest_start = time.monotonic()
        pipeline.run(progress_callback=progress_cb)
        ingest_elapsed = time.monotonic() - ingest_start
        tracker.finish()
        typer.echo(f"Ingested {len(all_input_files)} document(s) -> {sources_dir_path}")
        typer.echo("Run 'knox compile' to resolve, compile, and render the wiki.")

        base_dir = pipeline_config.resolved_base_dir()
        log_path = base_dir / "log.md"
        log_entry = generate_stage_log_entry(
            stage="ingest",
            duration=round(ingest_elapsed, 2),
            source_count=len(all_input_files),
            source_names=[Path(f).name for f in all_input_files],
            chunk_count=pipeline.last_chunk_count,
            llm_model=pipeline_config.llm.default_model or None,
            llm_usage=pipeline.llm_client.usage.to_dict() if pipeline.llm_client else None,
        )
        prepend_log(log_path, log_entry)
    except Exception as exc:
        tracker.finish()
        typer.echo(f"Ingest failed: {exc}", err=True)
        raise typer.Exit(1) from exc
