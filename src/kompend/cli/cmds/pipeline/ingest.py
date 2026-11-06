"""``kompend ingest`` — extract and cache sources without compiling."""

from __future__ import annotations

from kompend.cli.args import (
    CliSet,
    Ignore,
    IngestFiles,
    InPlaceFlag,
    OnlineUrls,
    Progress,
    RecursiveScan,
    ReparseFlag,
    SourceAttrs,
    SourceTags,
    UrlFile,
    Workers,
)
from kompend.cli.context import current_run_context
from kompend.cli.runtime.progress import PipelineProgress
from kompend.cli.settings import build_cli_overrides, config_file_from_ctx, parse_attrs, parse_cli_set, vault_paths_from_ctx
from kompend.cli.workflow.pipeline import resolve_compile_inputs
from kompend.core.cancel import CancelState
from kompend.core.config import resolve_config
from kompend.core.pipeline.ingest import ingest_documents
from kompend.core.trust.actor import resolve_actor
from kompend.logging import get_console, get_err_console
from kompend.output.pages.log import log_stage
from kompend.preprocessing.parsers import ParserOptions, build_parser


def ingest_cmd(
    files: IngestFiles = None,
    *,
    online_urls: OnlineUrls = None,
    url_file: UrlFile = None,
    ignore: Ignore = None,
    reparse: ReparseFlag = None,
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

    vp = vault_paths_from_ctx()
    no_user_tracking = current_run_context().no_user_tracking
    cli_overrides = build_cli_overrides(
        reparse=reparse,
        workers=workers,
        in_place_sources=in_place,
        no_user_tracking=no_user_tracking,
        cli_set=parse_cli_set(cli_set),
    )
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=cli_overrides, config_file=config_file_from_ctx())
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
        raise SystemExit(1)

    get_console().print(f"Ingesting {len(all_input_files)} document(s)")

    parser = build_parser(ParserOptions.from_config(cfg, base_dir=base_dir))
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
        result = ingest_documents(
            files=all_input_files,
            base_dir=base_dir,
            parser=parser,
            progress_fn=_progress,
            reparse=bool(cfg.kompend.compile.reparse or cfg.kompend.compile.no_cache),
            config=dict(cfg.flat()) if hasattr(cfg, "flat") else None,
            ingested_by=resolve_actor(cfg),
            tags=tags,
            attributes=parse_attrs(attrs),
            rumdl_config=cfg.kompend.compile.output.rumdl_config,
            flavor=cfg.kompend.compile.output.flavor,
            cancel=cancel,
            parse_timeout_s=cfg.kompend.ingest.parse_timeout_s,
        )
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
