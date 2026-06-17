"""Parse, compile, and ingest pipeline commands."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict

import typer

from knox.cli._pipeline_helpers import (
    _make_progress_callback,
    _post_compile_actions,
    _print_tier_listing,
    _resolve_compile_inputs,
)
from knox.cli._progress import PipelineProgress
from knox.cli._shared import build_cli_overrides, load_hooks, resolve_project_paths
from knox.cli.app import app
from knox.cli.args import (
    DEFAULT_COMPILE_FORMAT,
    DEFAULT_COMPILER,
    DEFAULT_IMAGE_MODE,
    DEFAULT_LLM_FALLBACK_MAX_RETRIES,
    DEFAULT_PROMPT_BUDGET,
    DEFAULT_START_AT_TIER,
    BudgetConcepts,
    BudgetGenSegment,
    BypassReview,
    CompileFiles,
    CompileLint,
    CompilerChoice,
    ConfigPath,
    EmbedBatchSize,
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
from knox.core.paths import KNOX_DIR
from knox.core.pipeline import build_pipeline
from knox.core.pipeline_defs import PipelineConfig
from knox.preprocessing.parsers._registry import build_default_router
from knox.preprocessing.parsers.base import ParserRouter

logger = logging.getLogger(__name__)


@app.command(name="parse")
def parse_cmd(
    files: ParseFiles,
    recursive: bool = False,
    print_elements: PrintElements = False,
) -> None:
    """Parse documents and show their structure.

    Accepts one or more file paths, glob patterns (e.g. *.pdf), or directories.
    """
    resolved = _resolve_compile_inputs(
        files=files,
        online_urls=[],
        url_file=None,
        sources_dir_path=None,  # type: ignore[arg-type]
        recursive=recursive,
        ignore=None,
    )
    router: ParserRouter = build_default_router()
    for path in resolved:
        parser = router.route(path)
        if parser is None:
            typer.echo(f"No parser for {path}", err=True)
            continue
        doc = parser.parse(path)
        typer.echo(f"{path.name}: {len(doc.elements)} element(s)")
        if print_elements:
            for el in doc.elements:
                typer.echo(f"  - {el.get('type')}: {str(el.get('text', ''))[:80]}")


@app.command(name="compile")
def compile_cmd(
    ctx: typer.Context,
    files: CompileFiles,
    config: ConfigPath = None,
    online_urls: OnlineUrls = None,
    url_file: UrlFile = None,
    recursive: bool = False,
    target_format: TargetFormat = DEFAULT_COMPILE_FORMAT,
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
    provenance: bool = False,
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
        wiki.metadata["source_names"] = [str(f) for f in all_input_files[:20]]

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
            show_progress=show_progress,
            llm_client=pipeline.llm_client,
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
    recursive: bool = False,
    image_mode: ImageMode = DEFAULT_IMAGE_MODE,
    ignore: Ignore = None,
    no_cache: NoCacheFlag = False,
    workers: Workers = 1,
) -> None:
    """Ingest documents into the knowledge graph without compiling a wiki.

    Parses every source and stores the structured documents in
    ``.knox/parsed/``. No chunking, no LLM — this is a fast,
    deterministic extraction pass that you can re-run anytime.

    Run ``knox compile`` afterwards to chunk, resolve, and compile
    the ingested documents into wiki pages.
    """
    if online_urls is None:
        online_urls = []

    cli_overrides = build_cli_overrides(
        image_mode=image_mode,
        no_cache=no_cache,
        workers=workers,
    )
    cfg = resolve_config(cli_overrides=cli_overrides, config_path=config)
    sources_dir_path, _output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)

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

    typer.echo(f"Ingesting {len(all_input_files)} document(s)")

    router: ParserRouter = build_default_router()
    parsed_dir = sources_dir_path / KNOX_DIR / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for path in all_input_files:
        parser = router.route(path)
        if parser is None:
            typer.echo(f"  Skipping {path.name} — no parser available", err=True)
            continue
        try:
            doc = parser.parse(path)
        except Exception:
            logger.exception("Failed to parse %s", path)
            typer.echo(f"  Failed to parse {path.name}", err=True)
            continue

        # Store parsed document as JSON alongside a content-hash key.
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        out_path = parsed_dir / f"{path.stem}_{content_hash}.json"
        out_path.write_text(
            json.dumps(asdict(doc), indent=2, default=str),
            encoding="utf-8",
        )
        count += 1

    typer.echo(f"Ingested {count} document(s) into {parsed_dir}")
