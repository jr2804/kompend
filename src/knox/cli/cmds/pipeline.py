"""Parse, compile, and ingest pipeline commands."""

from __future__ import annotations

import time
from pathlib import Path

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
    InPlaceFlag,
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
from knox.core.config.keys import COMPILE_IN_PLACE_SOURCES, PATHS_WIKI_DIR
from knox.core.ingest import ingest_documents
from knox.core.models import MetadataField, WikiField
from knox.core.paths import VaultPaths
from knox.core.pipeline import build_pipeline
from knox.core.pipeline_defs import PipelineConfig
from knox.core.user import resolve_actor
from knox.llm.errors import LLMError
from knox.logging import get_console, get_err_console, get_logger
from knox.output.wiki_index import generate_stage_log_entry, prepend_log
from knox.preprocessing.parsers._registry import build_default_router
from knox.preprocessing.parsers.base import ParserRouter

logger = get_logger(__name__)


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
        sources_dir=None,  # ty: ignore
        recursive=recursive,
        ignore=None,
    )
    router: ParserRouter = build_default_router()
    for path in resolved:
        parser = router.route(path)
        if parser is None:
            get_err_console().print(f"No parser for {path}")
            continue
        doc = parser.parse(path)
        get_console().print(f"{path.name}: {len(doc.elements)} element(s)")
        if print_elements:
            for el in doc.elements:  # ty: ignore
                get_console().print(f"  - {el.get('type')}: {str(el.get('text', ''))[:80]}")  # ty: ignore


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
    in_place: InPlaceFlag = False,
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
        in_place_sources=in_place,
    )
    cfg = resolve_config(cli_overrides=cli_overrides, config_path=config)
    sources_dir, output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)

    all_input_files = _resolve_compile_inputs(
        files,
        online_urls,
        url_file,
        sources_dir,
        recursive,
        ignore,
        in_place=cfg.get(COMPILE_IN_PLACE_SOURCES, False),
    )
    if not all_input_files:
        get_err_console().print(f"No supported files found in {sources_dir}.")
        raise typer.Exit(1)

    get_console().print(f"Compiling {len(all_input_files)} document(s) -> {output_dir}")

    pipeline_config = PipelineConfig.from_resolved(
        cfg,
        source_dir=sources_dir,
        output_dir=output_dir,
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
    # Start the bar eagerly (idempotent). The parse handler also calls start(),
    # but when parsing is skipped (--start-at-tier >= resolve) only this call
    # guarantees the live display is running before any phase.
    tracker.start()

    try:
        start_time = time.monotonic()
        wiki = pipeline.run(progress_callback=progress_cb)
        elapsed = time.monotonic() - start_time

        wiki.metadata[str(WikiField.DURATION_SECONDS)] = round(elapsed, 2)
        wiki.metadata[str(MetadataField.SOURCE_COUNT)] = len(all_input_files)
        wiki.metadata[str(MetadataField.SOURCE_NAMES)] = [str(f) for f in all_input_files[:20]]

        tracker.finish()
        get_console().print(f"Compiled wiki written to {output_dir}")

        _post_compile_actions(
            wiki=wiki,
            pipeline_config=pipeline_config,
            output_dir=output_dir,
            sources_dir=sources_dir,
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
        if isinstance(exc, LLMError):
            parts = [f"Error: {exc}"]
            if exc.model:
                parts.append(f"  model: {exc.model}")
            parts.append(f"  attempts: {exc.attempts}")
            if exc.cause:
                parts.append(f"  cause: {exc.cause}")
            get_err_console().print("\n".join(parts))
        else:
            get_err_console().print(f"Error: {exc}")
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
    in_place: InPlaceFlag = False,
    show_progress: Progress = None,
) -> None:
    """Ingest documents into the knowledge graph without compiling a wiki.

    Parses every source and caches the structured documents in
    ``.knox/cache/`` so ``knox compile`` can pick them up. No chunking,
    no LLM — this is a fast, deterministic extraction pass.

    Run ``knox compile`` afterwards to chunk, resolve, and compile
    the ingested documents into wiki pages.
    """
    if online_urls is None:
        online_urls = []

    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    no_user_tracking = ctx.obj.get("no_user_tracking") if ctx.obj else None
    cli_overrides = build_cli_overrides(
        wiki_dir=wiki_dir_override,
        image_mode=image_mode,
        no_cache=no_cache,
        workers=workers,
        in_place_sources=in_place,
        no_user_tracking=no_user_tracking,
    )
    cfg = resolve_config(cli_overrides=cli_overrides, config_path=config)
    sources_dir, _output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)
    base_dir = Path(cfg.get(PATHS_WIKI_DIR))

    all_input_files = _resolve_compile_inputs(
        files,
        online_urls,
        url_file,
        sources_dir,
        recursive,
        ignore,
        in_place=cfg.get(COMPILE_IN_PLACE_SOURCES, False),
    )
    if not all_input_files:
        get_err_console().print(f"No supported files found in {sources_dir}.")
        raise typer.Exit(1)

    get_console().print(f"Ingesting {len(all_input_files)} document(s)")

    images_dir = VaultPaths(base_dir).images_dir
    router: ParserRouter = build_default_router(images_dir=images_dir)
    tracker = PipelineProgress.create(
        total_files=len(all_input_files),
        force_progress=show_progress,
    )
    tracker.start()

    def _progress(current: int, total: int, detail: str) -> None:
        if detail == "done":
            tracker.update_progress(total, "Ingest complete")
        else:
            tracker.update_progress(current, f"[{current + 1}/{total}] {detail}")

    try:
        ingest_start = time.monotonic()
        result = ingest_documents(
            files=all_input_files,
            base_dir=base_dir,
            router=router,
            progress_fn=_progress,
            no_cache=no_cache,
            config=dict(cfg.flat()) if hasattr(cfg, "flat") else None,
            ingested_by=resolve_actor(cfg),
        )
        ingest_duration = time.monotonic() - ingest_start
    finally:
        tracker.finish()

    summary = f"Ingested {result.count} document(s) into {result.cache_dir}"
    if result.cached:
        summary += f" ({result.cached} cached)"
    if result.skipped:
        summary += f" ({result.skipped} skipped)"
    if result.failed:
        summary += f" ({result.failed} failed)"
    if result.dedup_skipped:
        summary += f" ({result.dedup_skipped} duplicates identified)"
    get_console().print(summary)

    log_file = VaultPaths(base_dir).log_file
    prepend_log(
        log_file,
        generate_stage_log_entry(
            stage="ingest",
            duration=ingest_duration,
            source_count=result.count,
            source_names=[f.stem for f in all_input_files],
            chunk_count=result.count,
        ),
    )
