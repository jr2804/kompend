"""Parse, compile, and ingest pipeline commands."""

from __future__ import annotations

import json
import sys
import time
import traceback
from typing import Any

import typer

from kompend.cli._distill_helpers import (
    count_pending_notes,
    distill_result_lines,
    pending_notes_message,
    run_distill,
)
from kompend.cli._pipeline_helpers import (
    PostCompileOptions,
    check_llm_availability,
    emit_json_done,
    make_json_progress_callback,
    make_progress_callback,
    post_compile_actions,
    print_tier_listing,
    resolve_compile_inputs,
)
from kompend.cli._post_compile import write_wiki_cache
from kompend.cli._progress import PipelineProgress
from kompend.cli._shared import build_cli_overrides, vault_paths_from_ctx
from kompend.cli._sources import resolve_files
from kompend.cli.app import app
from kompend.cli.args import (
    DEFAULT_START_AT_TIER,
    BypassReview,
    CliSet,
    CompileDistill,
    CompileLint,
    CompileLintAll,
    ConfigPath,
    EmbedModel,
    ForceTier,
    Ignore,
    IndexStore,
    IngestFiles,
    InPlaceFlag,
    JsonProgress,
    ListTiers,
    LlmDefaultModel,
    LlmFallbackChain,
    MemoryEnabled,
    NoCacheFlag,
    NoPrivacy,
    OnlineUrls,
    ParseFiles,
    PrintElements,
    Private,
    Progress,
    Provenance,
    RecursiveScan,
    ReviewMode,
    SourceAttrs,
    SourceTags,
    StartAtTier,
    SynthesizeFlag,
    UrlFile,
    Verify,
    VlmModel,
    WikiDir,
    Workers,
)
from kompend.core.cancel import CancelState, KompendCancelled
from kompend.core.config import DEFAULTS, coerce_config_value, resolve_config, validate_config_key
from kompend.core.config._configs import LintConfig
from kompend.core.config.keys import (
    DISPLAY_CLI_ELEMENT_CHARS,
)
from kompend.core.models import MetadataField, WikiField, load_cached_wiki, load_wiki_from_files
from kompend.core.paths import VaultPaths, vault_paths
from kompend.core.pipeline.config import PipelineConfig
from kompend.core.pipeline.ingest import ingest_documents
from kompend.core.pipeline.orchestrator import build_pipeline
from kompend.core.trust.actor import resolve_actor
from kompend.core.utils.exceptions import LLMOutputError
from kompend.logging import get_console, get_err_console, get_logger
from kompend.output.pages.log import log_stage
from kompend.preprocessing.parsers._registry import ParserBuildOptions, build_default_router
from kompend.preprocessing.parsers.base import ParserRouter
from kompend.rag import build_vector_index

logger = get_logger(__name__)


@app.command(name="parse")
def parse_cmd(
    ctx: typer.Context,
    files: ParseFiles,
    recursive: bool = False,
    print_elements: PrintElements = False,
) -> None:
    """Parse documents and show their structure.

    Accepts one or more file paths, glob patterns (e.g. *.pdf), or directories.
    """
    resolved = resolve_files(
        *files,
        recursive=recursive,
        ignore_patterns=None,
    )
    vp = vault_paths_from_ctx(ctx, require_vault=False)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())
    router: ParserRouter = build_default_router(
        ParserBuildOptions.from_config(cfg, images_dir=vp.images_dir),
    )
    for path in resolved:
        try:
            doc = router.parse(path)
        except ValueError as exc:
            # No backend claims the file, or every backend failed.
            get_err_console().print(f"No parser for {path}: {exc}")
            continue
        get_console().print(f"{path.name}: {len(doc.elements)} element(s)")
        if print_elements:
            element_text_limit = DEFAULTS.get(DISPLAY_CLI_ELEMENT_CHARS, 80)
            for el in doc.elements:
                content = el.content if isinstance(el.content, str) else repr(el.content)
                get_console().print(f"  - {el.type}: {content[:element_text_limit]}")


@app.command(name="compile")
def compile_cmd(  # noqa: PLR0915, PLR0917
    ctx: typer.Context,
    wiki_dir: WikiDir = None,
    config: ConfigPath = None,
    online_urls: OnlineUrls = None,
    url_file: UrlFile = None,
    embed_model: EmbedModel = None,
    ignore: Ignore = None,
    distill: CompileDistill = False,
    show_progress: Progress = None,
    start_at_tier: StartAtTier = DEFAULT_START_AT_TIER,
    no_cache: NoCacheFlag = None,
    workers: Workers = 1,
    force: ForceTier = None,
    in_place: InPlaceFlag = None,
    recursive: RecursiveScan = None,
    list_tiers: ListTiers = False,
    review: ReviewMode = None,
    bypass_review: BypassReview = False,
    provenance: Provenance = None,
    private: Private = None,
    no_privacy: NoPrivacy = None,
    verify: Verify = None,
    synthesize: SynthesizeFlag = None,
    index_store: IndexStore = None,
    compile_lint: CompileLint = True,
    compile_lint_all: CompileLintAll = True,
    llm_fallback_chain: LlmFallbackChain = None,
    llm_fallback_max_retries: LlmFallbackMaxRetries = DEFAULT_LLM_FALLBACK_MAX_RETRIES,
    llm_default_model: LlmDefaultModel = None,
    vlm_model: VlmModel = None,
    memory_enabled: MemoryEnabled = None,
    cli_set: CliSet = None,
    json_progress: JsonProgress = False,
) -> None:
    """Compile documents into a knowledge wiki.

    Reads all source files from the vault's ``sources/`` directory
    (configured via ``[paths] sources_dir`` in the project config).
    Use ``kompend ingest`` to add new files or URLs to the vault first.

    Pass ``--distill`` to merge captured/learned notes into
    ``sources/_distilled/`` pages (archiving the notes it merged) before
    compiling — otherwise notes awaiting distillation are only reported.

    Output shape (format, flavor, frontmatter, kind, compiler, image_mode) and
    tuning values (budgets, timeouts, decay params) are read from config —
    override per-run with ``--set compile.<key>=<val>`` or permanently with
    ``kompend config set``.
    """
    if online_urls is None:
        online_urls = []

    vp = vault_paths_from_ctx(ctx)
    if wiki_dir:
        vp = vault_paths(wiki_dir)
    cli_overrides = build_cli_overrides(
        start_at_tier=start_at_tier,
        no_cache=no_cache or False,
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
        ignore=ignore,
        llm_default_model=llm_default_model,
        vlm_model=vlm_model,
        memory_enabled=memory_enabled,
        llm_fallback_chain=llm_fallback_chain,
        embed_model=embed_model,
        in_place_sources=in_place,
        recursive=recursive,
        cli_set=parse_cli_set(cli_set),
    )
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=cli_overrides, config_file=config)
    sources_dir = vp.sources_dir
    output_dir = vp.output_dir

    console = get_err_console() if json_progress else get_console()

    pending_notes = count_pending_notes(sources_dir)
    if distill and not list_tiers:
        if pending_notes:
            console.print(f"Distilling {pending_notes} note(s) before compile…")
        for line in distill_result_lines(run_distill(sources_dir, cfg), sources_dir):
            console.print(line)
    elif pending_notes and not list_tiers:
        console.print(pending_notes_message(pending_notes))

    pipeline_config = PipelineConfig.from_resolved(
        cfg,
        source_dir=sources_dir,
        output_dir=output_dir,
        input_files=None,
    )

    cancel = CancelState()
    pipeline = build_pipeline(pipeline_config, cancel=cancel)

    source_files = pipeline.iter_sources()
    if not source_files:
        get_err_console().print(f"No supported files found in {sources_dir}.")
        raise typer.Exit(1)

    console.print(f"Compiling {len(source_files)} document(s) -> {output_dir}")

    check_llm_availability(pipeline)

    if list_tiers:
        print_tier_listing(pipeline)
        return

    if json_progress:
        progress_cb = make_json_progress_callback()
    else:
        tracker = PipelineProgress.create(
            total_files=len(source_files),
            force_progress=show_progress,
            cancel=cancel,
        )
        progress_cb = make_progress_callback(tracker) if tracker.enabled else None
        # Start the bar eagerly (idempotent). The parse handler also calls start(),
        # but when parsing is skipped (--start-at-tier >= resolve) only this call
        # guarantees the live display is running before any phase.
        tracker.start()

    try:
        start_time = time.monotonic()
        wiki = pipeline.run(progress_callback=progress_cb)
        elapsed = time.monotonic() - start_time

        wiki.metadata[str(WikiField.DURATION_SECONDS)] = round(elapsed, 2)
        wiki.metadata[str(MetadataField.SOURCE_COUNT)] = len(source_files)
        wiki.metadata[str(MetadataField.SOURCE_NAMES)] = [str(f) for f in source_files[: cfg.kompend.display.source_names]]

        if json_progress:
            emit_json_done(elapsed, f"Compiled wiki written to {output_dir}")
            console = get_err_console()
        else:
            tracker.finish()
            console = get_console()
        console.print(f"Compiled wiki written to {output_dir}")

        lint_config = LintConfig(
            fuzzy_threshold=float(cfg.kompend.lint.fuzzy_threshold),
        )
        post_compile_actions(
            PostCompileOptions(
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
                compile_lint_all=compile_lint_all,
                embed_model=embed_model,
                show_progress=show_progress,
                lint_config=lint_config,
                embed_attachments=cfg.kompend.attachments.embed,
            )
        )

    except KompendCancelled:
        # Graceful user abort (double-ESC): pipeline.run() already saved tier
        # state and unwound cleanly — stop the bar and re-raise for the
        # top-level handler (exit 130, no traceback).
        if not json_progress:
            tracker.stop()
        raise

    except Exception as exc:
        if not json_progress:
            tracker.stop()
        if isinstance(exc, LLMOutputError):
            err_msg = f"Error: {exc}"
        else:
            err_msg = f"Error: {exc}"
            get_err_console().print(traceback.format_exc())

        if json_progress:
            err_payload = json.dumps({"event": "error", "message": err_msg}, ensure_ascii=False)
            sys.stdout.write(err_payload + "\n")
            sys.stdout.flush()
        else:
            get_err_console().print(err_msg)
        raise typer.Exit(1) from exc


@app.command(name="ingest")
def ingest_cmd(  # noqa: PLR0917 — Typer command with many options
    ctx: typer.Context,
    files: IngestFiles = None,
    config: ConfigPath = None,
    online_urls: OnlineUrls = None,
    url_file: UrlFile = None,
    ignore: Ignore = None,
    no_cache: NoCacheFlag = None,
    workers: Workers = 1,
    in_place: InPlaceFlag = None,
    recursive: RecursiveScan = None,
    show_progress: Progress = None,
    cli_set: CliSet = None,
    tags: SourceTags = None,
    attrs: SourceAttrs = None,
) -> None:
    """Ingest documents into the knowledge graph without compiling a wiki.

    Parses every source and caches the structured documents in
    ``.kompend/cache/`` so ``kompend compile`` can pick them up. No chunking,
    no LLM — this is a fast, deterministic extraction pass.

    Run ``kompend compile`` afterwards to chunk, resolve, and compile
    the ingested documents into wiki pages.
    """
    if online_urls is None:
        online_urls = []

    vp = vault_paths_from_ctx(ctx)
    no_user_tracking = ctx.obj.get("no_user_tracking") if ctx.obj else None
    cli_overrides = build_cli_overrides(
        no_cache=no_cache or False,
        workers=workers,
        in_place_sources=in_place,
        no_user_tracking=no_user_tracking,
        cli_set=parse_cli_set(cli_set),
    )
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=cli_overrides, config_file=config)
    sources_dir = vp.sources_dir
    base_dir = vp.base_dir

    all_input_files = resolve_compile_inputs(
        files or [],
        online_urls,
        url_file,
        sources_dir,
        cfg.kompend.paths.recursive if recursive is None else recursive,
        ignore,
        in_place=cfg.kompend.compile.in_place_sources,
        undetected=cfg.kompend.downloader.undetected,
    )
    if not all_input_files:
        get_err_console().print(f"No supported files found in {sources_dir}.")
        raise typer.Exit(1)

    get_console().print(f"Ingesting {len(all_input_files)} document(s)")

    images_dir = VaultPaths(base_dir).images_dir
    router: ParserRouter = build_default_router(
        ParserBuildOptions.from_config(cfg, images_dir=images_dir),
    )
    cancel = CancelState()
    tracker = PipelineProgress.create(
        total_files=len(all_input_files),
        force_progress=show_progress,
        cancel=cancel,
    )
    tracker.start()

    phase = 0

    def _progress(current: int, total: int, detail: str) -> None:
        nonlocal phase
        if detail == "done":
            tracker.update_progress(total, "Ingest complete")
        elif detail.startswith("embedding "):
            if phase == 0:
                phase = 1
                tracker.reset_for_phase(total, f"Embedding {total} document(s)")
            tracker.update_progress(current, f"[{current + 1}/{total}] {detail}")
        else:
            tracker.update_progress(current, f"[{current + 1}/{total}] {detail}")

    try:
        ingest_start = time.monotonic()
        result = ingest_documents(
            files=all_input_files,
            base_dir=base_dir,
            router=router,
            progress_fn=_progress,
            no_cache=no_cache or False,
            config=dict(cfg.flat()) if hasattr(cfg, "flat") else None,
            ingested_by=resolve_actor(cfg),
            tags=tags,
            attributes=_parse_attrs(attrs),
            rumdl_config=cfg.kompend.compile.output.rumdl_config,
            flavor=cfg.kompend.compile.output.flavor,
            cancel=cancel,
        )
        time.monotonic() - ingest_start
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
    log_stage(vp, action="Ingest", detail=summary)


@app.command(name="index")
def index_cmd(
    ctx: typer.Context,
    embed_model: EmbedModel = None,
    config: ConfigPath = None,
    wiki_dir: WikiDir = None,
    cli_set: CliSet = None,
) -> None:
    """Rebuild the zvec vector index for semantic search.

    Uses the cached :class:`CompiledWiki` from the ``wiki_snapshot`` table in ``.kompend/kompend.db``.
    This is a recovery command — a normal ``kompend compile`` already builds
    the index automatically via post-compile actions.

    Use it when the index is missing, stale, or you changed the embedding
    model (``compile.embedding.model``) or the ``attachments.embed``
    setting and want to re-embed without a full LLM recompile.

    It does **not** re-ingest attachments or recompile the wiki. To ingest
    newly-added attachments, run ``kompend compile --force --start-at-tier chunk``
    first.
    """
    vp = vault_paths_from_ctx(ctx)
    if wiki_dir:
        vp = vault_paths(wiki_dir)

    cli_overrides = build_cli_overrides(cli_set=parse_cli_set(cli_set))
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=cli_overrides, config_file=config)
    output_dir = vp.output_dir

    # Index the markdown bundle — the node set ``kompend query`` judges staleness
    # against (a files-loaded wiki carries every page; the snapshot graph may
    # not, e.g. per-source summaries).
    cached = load_cached_wiki(vp.base_dir)
    wiki = load_wiki_from_files(output_dir) or cached
    if wiki is None:
        get_err_console().print(f"No compiled wiki found in {output_dir}. Run 'kompend compile' first.")
        raise typer.Exit(1)

    # Resolve embed model (CLI flag > config > compile.embedding.model).
    resolved_model = embed_model or cfg.kompend.compile.embedding.model
    if not resolved_model:
        get_err_console().print(
            "No embedding model configured. Set compile.embedding.model in config or pass --embed-model.",
        )
        raise typer.Exit(1)

    embed_attachments = cfg.kompend.attachments.embed

    try:
        count = build_vector_index(
            wiki=wiki,
            output_dir=output_dir,
            embed_model=resolved_model,
            base_dir=vp.base_dir,
            embed_attachments=embed_attachments,
        )
    except Exception as exc:
        get_err_console().print(f"Failed to build vector index: {exc}")
        raise typer.Exit(1) from exc

    # Persist embedding model metadata so ``kompend query`` can open the store.
    # Written through the snapshot wiki, which is what the cache holds (the
    # files-loaded wiki above has no graph metrics to cache).
    if cached is not None:
        cached.metadata[str(MetadataField.EMBEDDING_MODEL)] = resolved_model
        write_wiki_cache(cached, vp.base_dir)

    if count > 0:
        get_console().print(f"Indexed {count} nodes in zvec store for semantic search.")
    else:
        get_err_console().print("Warning: no nodes indexed — vector index is empty.")


def parse_cli_set(items: list[str] | None) -> dict[str, Any]:
    """Parse repeated ``--set key=value`` items into a coerced override dict.

    Each value is validated against the known config schema and coerced to the
    type declared in :data:`kompend.core.config.DEFAULTS`. Unknown keys or bad
    values abort the run with a clear error. The result is merged last by
    :func:`build_cli_overrides` so it overrides every other CLI flag.
    """
    result: dict[str, Any] = {}
    if not items:
        return result
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"--set expects key=value, got: {item!r}")
        key, _, val = item.partition("=")
        key = key.strip()
        if not validate_config_key(key):
            raise typer.BadParameter(f"unknown config key: {key!r}")
        try:
            result[key] = coerce_config_value(val, key)
        except ValueError as exc:
            raise typer.BadParameter(f"--set {key}: {exc}") from exc
    return result


def _parse_attrs(items: list[str] | None) -> dict[str, str]:
    """Parse repeated ``--attr key=value`` items into a plain string dict."""
    result: dict[str, str] = {}
    if not items:
        return result
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"--attr expects key=value, got: {item!r}")
        key, _, val = item.partition("=")
        result[key.strip()] = val
    return result
