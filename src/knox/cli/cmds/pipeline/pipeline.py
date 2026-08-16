"""Parse, compile, and ingest pipeline commands."""

from __future__ import annotations

import json
import sys
import time
import traceback
from typing import Annotated, Any

import typer

from knox.cli._pipeline_helpers import (
    PostCompileOptions,
    check_llm_availability,
    emit_json_done,
    make_json_progress_callback,
    make_progress_callback,
    post_compile_actions,
    print_tier_listing,
    resolve_compile_inputs,
)
from knox.cli._post_compile import write_wiki_cache
from knox.cli._progress import PipelineProgress
from knox.cli._shared import build_cli_overrides, vault_paths_from_ctx
from knox.cli._sources import resolve_files
from knox.cli.app import app
from knox.cli.args import (
    DEFAULT_LLM_FALLBACK_MAX_RETRIES,
    DEFAULT_START_AT_TIER,
    BypassReview,
    CliSet,
    CompileLint,
    ConfigPath,
    EmbedModel,
    ForceTier,
    Ignore,
    IndexStore,
    InPlaceFlag,
    JsonProgress,
    ListTiers,
    LlmDefaultModel,
    LlmFallbackChain,
    LlmFallbackMaxRetries,
    LlmRoleModel,
    MemoryEnabled,
    NoCacheFlag,
    NoPrivacy,
    OnlineUrls,
    ParseFiles,
    Parser,
    PrintElements,
    Private,
    Progress,
    Provenance,
    ReviewMode,
    SourceTags,
    StartAtTier,
    SynthesizeFlag,
    UrlFile,
    Verify,
    VlmModel,
    WikiDir,
    Workers,
)
from knox.core.config import DEFAULTS, coerce_config_value, resolve_config, validate_config_key
from knox.core.config._configs import LintConfig
from knox.core.config.keys import (
    COMPILE_EMBEDDING_MODEL,
    COMPILE_IN_PLACE_SOURCES,
    DISPLAY_CLI_ELEMENT_CHARS,
    DISPLAY_SOURCE_NAMES,
    DOWNLOADER_UNDETECTED,
    INGEST_PARSER,
    LINT_FUZZY_THRESHOLD,
    SEARCH_EMBED_ATTACHMENTS,
)
from knox.core.models import MetadataField, WikiField, load_cached_wiki
from knox.core.paths import VaultPaths, vault_paths
from knox.core.pipeline.config import PipelineConfig
from knox.core.pipeline.ingest import ingest_documents
from knox.core.pipeline.orchestrator import build_pipeline
from knox.core.trust.actor import resolve_actor
from knox.core.utils.exceptions import LLMOutputError
from knox.logging import get_console, get_err_console, get_logger
from knox.output.pages.log import log_stage
from knox.preprocessing.parsers._registry import ParserBuildOptions, build_default_router, get_all_extensions
from knox.preprocessing.parsers.base import ParserRouter
from knox.rag import build_vector_index

logger = get_logger(__name__)


@app.command(name="parse")
def parse_cmd(
    files: ParseFiles,
    recursive: bool = False,
    print_elements: PrintElements = False,
    parser: Parser = None,
) -> None:
    """Parse documents and show their structure.

    Accepts one or more file paths, glob patterns (e.g. *.pdf), or directories.
    """
    resolved = resolve_files(
        *files,
        recursive=recursive,
        ignore_patterns=None,
    )
    router: ParserRouter = build_default_router(names=parser or "liteparse")
    for path in resolved:
        instance = router.route(path)
        if instance is None:
            get_err_console().print(f"No parser for {path}")
            continue
        doc = instance.parse(path)
        get_console().print(f"{path.name}: {len(doc.elements)} element(s)")
        if print_elements:
            element_text_limit = DEFAULTS.get(DISPLAY_CLI_ELEMENT_CHARS, 80)
            for el in doc.elements:
                content = el.content if isinstance(el.content, str) else repr(el.content)
                get_console().print(f"  - {el.type}: {content[:element_text_limit]}")


@app.command(name="compile")
def compile_cmd(  # noqa: PLR0915
    ctx: typer.Context,
    wiki_dir: WikiDir = None,
    config: ConfigPath = None,
    online_urls: OnlineUrls = None,
    url_file: UrlFile = None,
    embed_model: EmbedModel = None,
    ignore: Ignore = None,
    show_progress: Progress = None,
    start_at_tier: StartAtTier = DEFAULT_START_AT_TIER,
    no_cache: NoCacheFlag = None,
    workers: Workers = 1,
    force: ForceTier = None,
    in_place: InPlaceFlag = None,
    list_tiers: ListTiers = False,
    review: ReviewMode = None,
    bypass_review: BypassReview = False,
    provenance: Provenance = None,
    private: Private = None,
    no_privacy: NoPrivacy = None,
    verify: Verify = None,
    synthesize: SynthesizeFlag = None,
    index_store: IndexStore = None,
    compile_lint: CompileLint = None,
    llm_fallback_chain: LlmFallbackChain = None,
    llm_fallback_max_retries: LlmFallbackMaxRetries = DEFAULT_LLM_FALLBACK_MAX_RETRIES,
    llm_default_model: LlmDefaultModel = None,
    vlm_model: VlmModel = None,
    memory_enabled: MemoryEnabled = None,
    llm_role_model: LlmRoleModel = None,
    cli_set: CliSet = None,
    json_progress: JsonProgress = False,
) -> None:
    """Compile documents into a knowledge wiki.

    Reads all source files from the vault's ``sources/`` directory
    (configured via ``[paths] sources_dir`` in the project config).
    Use ``knox ingest`` to add new files or URLs to the vault first.

    Output shape (format, flavor, frontmatter, kind, compiler, image_mode) and
    tuning values (budgets, timeouts, decay params) are read from config —
    override per-run with ``--set compile.<key>=<val>`` or permanently with
    ``knox config set``.
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
        llm_fallback_max_retries=llm_fallback_max_retries,
        llm_role_model=llm_role_model,
        embed_model=embed_model,
        in_place_sources=in_place,
        cli_set=parse_cli_set(cli_set),
    )
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=cli_overrides, config_file=config)
    sources_dir = vp.sources_dir
    output_dir = vp.output_dir

    source_extensions = get_all_extensions()
    source_files = [f for f in sources_dir.rglob("*") if f.suffix.lower() in source_extensions]
    if not source_files:
        get_err_console().print(f"No supported files found in {sources_dir}.")
        raise typer.Exit(1)

    get_console().print(f"Compiling {len(source_files)} document(s) -> {output_dir}")

    pipeline_config = PipelineConfig.from_resolved(
        cfg,
        source_dir=sources_dir,
        output_dir=output_dir,
        input_files=None,
    )

    pipeline = build_pipeline(pipeline_config)

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
        wiki.metadata[str(MetadataField.SOURCE_NAMES)] = [str(f) for f in source_files[: cfg.get(DISPLAY_SOURCE_NAMES, 20)]]

        if json_progress:
            emit_json_done(elapsed, f"Compiled wiki written to {output_dir}")
            console = get_err_console()
        else:
            tracker.finish()
            console = get_console()
        console.print(f"Compiled wiki written to {output_dir}")

        lint_config = LintConfig(
            fuzzy_threshold=float(cfg.get(LINT_FUZZY_THRESHOLD, 0.85)),
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
                embed_model=embed_model,
                show_progress=show_progress,
                lint_config=lint_config,
                embed_attachments=bool(cfg.get(SEARCH_EMBED_ATTACHMENTS, False)),
            )
        )

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
def ingest_cmd(
    ctx: typer.Context,
    files: Annotated[
        list[str] | None,
        typer.Argument(help="Files, glob patterns, or directories to ingest. If omitted, all supported files in the wiki sources dir are scanned."),
    ] = None,
    config: ConfigPath = None,
    online_urls: OnlineUrls = None,
    url_file: UrlFile = None,
    recursive: bool = False,
    ignore: Ignore = None,
    no_cache: NoCacheFlag = None,
    workers: Workers = 1,
    in_place: InPlaceFlag = None,
    show_progress: Progress = None,
    parser: Parser = None,
    cli_set: CliSet = None,
    tags: SourceTags = None,
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

    vp = vault_paths_from_ctx(ctx)
    no_user_tracking = ctx.obj.get("no_user_tracking") if ctx.obj else None
    cli_overrides = build_cli_overrides(
        no_cache=no_cache or False,
        workers=workers,
        in_place_sources=in_place,
        no_user_tracking=no_user_tracking,
        parser=parser,
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
        recursive,
        ignore,
        in_place=cfg.get(COMPILE_IN_PLACE_SOURCES, False),
        undetected=cfg.get(DOWNLOADER_UNDETECTED, True),
    )
    if not all_input_files:
        get_err_console().print(f"No supported files found in {sources_dir}.")
        raise typer.Exit(1)

    get_console().print(f"Ingesting {len(all_input_files)} document(s)")

    images_dir = VaultPaths(base_dir).images_dir
    parser_name = cfg.get(INGEST_PARSER)
    router: ParserRouter = build_default_router(ParserBuildOptions(images_dir=images_dir), names=parser_name)
    tracker = PipelineProgress.create(
        total_files=len(all_input_files),
        force_progress=show_progress,
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

    Uses the cached :class:`CompiledWiki` from ``.knox/last_wiki.json``.
    This is a recovery command — a normal ``knox compile`` already builds
    the index automatically via post-compile actions.

    Use it when the index is missing, stale, or you changed the embedding
    model (``compile.embedding.model``) or the ``search.embed_attachments``
    setting and want to re-embed without a full LLM recompile.

    It does **not** re-ingest attachments or recompile the wiki. To ingest
    newly-added attachments, run ``knox compile --force --start-at-tier chunk``
    first.
    """
    vp = vault_paths_from_ctx(ctx)
    if wiki_dir:
        vp = vault_paths(wiki_dir)

    cli_overrides = build_cli_overrides(cli_set=parse_cli_set(cli_set))
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=cli_overrides, config_file=config)
    output_dir = vp.output_dir

    wiki = load_cached_wiki(vp.base_dir)
    if wiki is None:
        get_err_console().print(f"No compiled wiki found in {output_dir}. Run 'knox compile' first.")
        raise typer.Exit(1)

    # Resolve embed model (CLI flag > config > compile.embedding.model).
    resolved_model = embed_model or cfg.get(COMPILE_EMBEDDING_MODEL, "")
    if not resolved_model:
        get_err_console().print(
            "No embedding model configured. Set compile.embedding.model in config or pass --embed-model.",
        )
        raise typer.Exit(1)

    embed_attachments = bool(cfg.get(SEARCH_EMBED_ATTACHMENTS, False))

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

    # Persist embedding model metadata so ``knox query`` can open the store.
    wiki.metadata[str(MetadataField.EMBEDDING_MODEL)] = resolved_model
    write_wiki_cache(wiki, vp.base_dir)

    if count > 0:
        get_console().print(f"Indexed {count} nodes in zvec store for semantic search.")
    else:
        get_err_console().print("Warning: no nodes indexed — vector index is empty.")


def parse_cli_set(items: list[str] | None) -> dict[str, Any]:
    """Parse repeated ``--set key=value`` items into a coerced override dict.

    Each value is validated against the known config schema and coerced to the
    type declared in :data:`knox.core.config.DEFAULTS`. Unknown keys or bad
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
