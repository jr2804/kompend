"""Parse, compile, and ingest pipeline commands."""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import typer

from knox.cli._pipeline_helpers import (
    _emit_json_done,
    _make_json_progress_callback,
    _make_progress_callback,
    _post_compile_actions,
    _print_tier_listing,
    _resolve_compile_inputs,
)
from knox.cli._progress import PipelineProgress
from knox.cli._shared import build_cli_overrides, resolve_project_paths
from knox.cli.app import app
from knox.cli.args import (
    DEFAULT_LLM_FALLBACK_MAX_RETRIES,
    DEFAULT_START_AT_TIER,
    BypassReview,
    CliSet,
    CompileFiles,
    CompileLint,
    ConfigPath,
    EmbedModel,
    ForceTier,
    Ignore,
    IndexStore,
    InPlaceFlag,
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
    StartAtTier,
    SynthesizeFlag,
    UrlFile,
    Verify,
    VlmModel,
    Workers,
)
from knox.core.config import coerce_config_value, resolve_config, validate_config_key
from knox.core.config.keys import COMPILE_IN_PLACE_SOURCES, DOWNLOADER_UNDETECTED, INGEST_PARSER, PATHS_WIKI_DIR
from knox.core.ingest import ingest_documents
from knox.core.models import MetadataField, WikiField
from knox.core.paths import VaultPaths
from knox.core.pipeline import build_pipeline
from knox.core.pipeline_defs import PipelineConfig
from knox.core.user import resolve_actor
from knox.llm.errors import LLMError
from knox.llm.roles import LLMRole
from knox.logging import get_console, get_err_console, get_logger
from knox.preprocessing.parsers._registry import ParserBuildOptions, build_default_router
from knox.preprocessing.parsers.base import ParserRouter

logger = get_logger(__name__)


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
    resolved = _resolve_compile_inputs(
        files=files,
        online_urls=[],
        url_file=None,
        sources_dir=None,  # ty: ignore
        recursive=recursive,
        ignore=None,
    )
    router: ParserRouter = build_default_router(names=parser or "docling")
    for path in resolved:
        instance = router.route(path)
        if instance is None:
            get_err_console().print(f"No parser for {path}")
            continue
        doc = instance.parse(path)
        get_console().print(f"{path.name}: {len(doc.elements)} element(s)")
        if print_elements:
            for el in doc.elements:  # ty: ignore
                get_console().print(f"  - {el.get('type')}: {str(el.get('text', ''))[:80]}")  # ty: ignore


@app.command(name="compile")
def compile_cmd(  # noqa: PLR0915
    ctx: typer.Context,
    config: ConfigPath = None,
    online_urls: OnlineUrls = None,
    url_file: UrlFile = None,
    recursive: bool = False,
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
    json_progress: bool = typer.Option(False, "--json-progress", help="Emit progress as JSONL to stdout (for machine consumers)"),
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

    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cli_overrides = build_cli_overrides(
        wiki_dir=wiki_dir_override,
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
    cfg = resolve_config(cli_overrides=cli_overrides, config_path=config)
    sources_dir, output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)

    source_extensions = ("*.pdf", "*.PDF", "*.docx", "*.DOCX", "*.doc", "*.DOC", "*.epub", "*.md", "*.txt", "*.html", "*.pptx")
    source_files = [f for ext in source_extensions for f in sources_dir.rglob(ext)]
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

    # Pre-flight LLM connectivity check — fail fast before processing documents.
    # Catches wrong provider prefixes, missing API keys, unreachable endpoints,
    # and invalid model names before the pipeline starts grinding through all
    # files with silent failures on every LLM call.
    if pipeline.llm_client is not None:
        try:
            ping = pipeline.llm_client.try_complete(
                [{"role": "user", "content": "Respond with exactly one word: ok"}],
                role=LLMRole.DOCTOR,
                max_tokens=128,
            )
        except Exception as exc:
            get_err_console().print("[red]LLM not available — not starting compilation.[/red]")
            get_err_console().print(f"  {exc}")
            raise typer.Exit(1) from exc
        if ping is None:
            report = pipeline.llm_client.degradation_report()
            recent = report.get("recent_failures", [])
            msg = recent[-1].get("message", "LLM is not responding") if recent else "LLM is not responding"
            get_err_console().print("[red]LLM not available — not starting compilation.[/red]")
            get_err_console().print(f"  Model: {pipeline.llm_client.default_model or '?'}")
            get_err_console().print(f"  Error: {msg}")
            raise typer.Exit(1)

    if list_tiers:
        _print_tier_listing(pipeline)
        return

    if json_progress:
        progress_cb = _make_json_progress_callback()
    else:
        tracker = PipelineProgress.create(
            total_files=len(source_files),
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
        wiki.metadata[str(MetadataField.SOURCE_COUNT)] = len(source_files)
        wiki.metadata[str(MetadataField.SOURCE_NAMES)] = [str(f) for f in source_files[:20]]

        if json_progress:
            _emit_json_done(elapsed, f"Compiled wiki written to {output_dir}")
            console = get_err_console()
        else:
            tracker.finish()
            console = get_console()
        console.print(f"Compiled wiki written to {output_dir}")

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
        if not json_progress:
            tracker.stop()
        if isinstance(exc, LLMError):
            parts = [f"Error: {exc}"]
            if exc.model:
                parts.append(f"  model: {exc.model}")
            parts.append(f"  attempts: {exc.attempts}")
            if exc.cause:
                parts.append(f"  cause: {exc.cause}")
            err_msg = "\n".join(parts)
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
    files: CompileFiles,
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
        no_cache=no_cache,
        workers=workers,
        in_place_sources=in_place,
        no_user_tracking=no_user_tracking,
        parser=parser,
        cli_set=parse_cli_set(cli_set),
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
            no_cache=no_cache,
            config=dict(cfg.flat()) if hasattr(cfg, "flat") else None,
            ingested_by=resolve_actor(cfg),
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
