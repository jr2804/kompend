"""``kompend compile`` — run the pipeline that writes the wiki."""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from kompend.cli.args import (
    DEFAULT_START_AT_TIER,
    BypassReview,
    CliSet,
    CompileDistill,
    CompileLint,
    CompileLintAll,
    EmbedModel,
    ForceTier,
    Ignore,
    IgnoreModelFlag,
    IndexStore,
    InPlaceFlag,
    JsonProgress,
    LintQuiet,
    LintVerbose,
    ListTiers,
    LlmDefaultModel,
    LlmFallbackChain,
    MemoryEnabled,
    NoPrivacy,
    OnlineUrls,
    Private,
    Progress,
    Provenance,
    RecursiveScan,
    ReparseFlag,
    ReviewMode,
    StartAtTier,
    SynthesizeFlag,
    UrlFile,
    Verify,
    VlmModel,
    WikiDir,
    Workers,
)
from kompend.cli.runtime.progress import PipelineProgress
from kompend.cli.settings import (
    build_cli_overrides,
    config_file_from_ctx,
    parse_cli_set,
    resolve_lint_output,
    vault_paths_from_ctx,
)
from kompend.cli.workflow.distill import count_pending_notes, distill_result_lines, pending_notes_message, run_distill
from kompend.cli.workflow.pipeline import (
    check_llm_availability,
    emit_json_done,
    make_json_progress_callback,
    make_progress_callback,
    print_tier_listing,
)
from kompend.cli.workflow.post_compile import PostCompileOptions, post_compile_actions
from kompend.core.cancel import CancelState, KompendCancelled
from kompend.core.config import ResolvedConfig, resolve_config
from kompend.core.config._configs import LintConfig
from kompend.core.models import CompiledWiki, MetadataField, WikiField
from kompend.core.paths import VaultPaths, vault_paths
from kompend.core.pipeline.config import PipelineConfig
from kompend.core.pipeline.orchestrator import WikiPipeline, build_pipeline
from kompend.core.utils.exceptions import LLMOutputError
from kompend.logging import get_console, get_err_console

ProgressCallback = Callable[[str, int, int, str], None]


def compile_cmd(
    *,
    wiki_dir: WikiDir = None,
    online_urls: OnlineUrls = None,
    url_file: UrlFile = None,
    embed_model: EmbedModel = None,
    ignore: Ignore = None,
    distill: CompileDistill = False,
    show_progress: Progress = None,
    start_at_tier: StartAtTier = DEFAULT_START_AT_TIER,
    reparse: ReparseFlag = None,
    ignore_model: IgnoreModelFlag = None,
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
    verbose_lint: LintVerbose = False,
    quiet_lint: LintQuiet = False,
    llm_fallback_chain: LlmFallbackChain = None,
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
    vp = vault_paths_from_ctx()
    if wiki_dir:
        vp = vault_paths(wiki_dir)
    cfg = resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(
            start_at_tier=start_at_tier,
            reparse=reparse,
            ignore_model=ignore_model,
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
        ),
        config_file=config_file_from_ctx(),
    )
    sources_dir = vp.sources_dir
    output_dir = vp.output_dir

    console = get_err_console() if json_progress else get_console()
    _publish_pending_notes(console, vp, cfg, distill=distill, list_tiers=list_tiers)

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
        raise SystemExit(1)

    console.print(f"Compiling {len(source_files)} document(s) -> {output_dir}")

    check_llm_availability(pipeline)

    if list_tiers:
        print_tier_listing(pipeline)
        return

    progress_cb, tracker = _start_progress(
        json_progress=json_progress,
        show_progress=show_progress,
        cancel=cancel,
        total_files=len(source_files),
    )

    try:
        wiki, elapsed = _run_pipeline(pipeline, progress_cb=progress_cb, output_dir=output_dir, console=console, tracker=tracker)
        wiki.metadata[str(WikiField.DURATION_SECONDS)] = round(elapsed, 2)
        wiki.metadata[str(MetadataField.SOURCE_COUNT)] = len(source_files)
        wiki.metadata[str(MetadataField.SOURCE_NAMES)] = [str(f) for f in source_files[: cfg.kompend.display.source_names]]

        _post_compile(
            wiki=wiki,
            vp=vp,
            cfg=cfg,
            pipeline_config=pipeline_config,
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
            verbose_lint=verbose_lint,
            quiet_lint=quiet_lint,
        )

    except KompendCancelled:
        # Graceful user abort (double-ESC): pipeline.run() already saved tier
        # state and unwound cleanly — stop the bar and re-raise for the
        # top-level handler (exit 130, no traceback).
        if tracker is not None:
            tracker.stop()
        raise

    except Exception as exc:
        _report_failure(exc, json_progress=json_progress, tracker=tracker)
        raise SystemExit(1) from exc


def _publish_pending_notes(
    console: Console,
    vp: VaultPaths,
    cfg: ResolvedConfig,
    *,
    distill: bool,
    list_tiers: bool,
) -> None:
    """Distill the captured notes, or say how many wait for it.

    ``--list-tiers`` is a read-only query, so it never moves files.
    """
    pending_notes = count_pending_notes(vp.sources_dir)
    if list_tiers:
        return
    if distill:
        if pending_notes:
            console.print(f"Distilling {pending_notes} note(s) before compile…")
        for line in distill_result_lines(run_distill(vp.sources_dir, cfg, memory_dir=vp.agent_memory_dir), vp.sources_dir):
            console.print(line)
    elif pending_notes:
        console.print(pending_notes_message(pending_notes))


def _start_progress(
    *,
    json_progress: bool,
    show_progress: Progress,
    cancel: CancelState,
    total_files: int,
) -> tuple[ProgressCallback | None, PipelineProgress | None]:
    """Start the display a compile run reports through.

    ``--json-progress`` owns stdout with one object per event, so the bar is
    silent and the callback is ``None``. Otherwise the bar starts eagerly
    (idempotent): when parsing is skipped (``--start-at-tier`` at or past
    resolve) only this call guarantees the live display runs before any phase.
    """
    if json_progress:
        return make_json_progress_callback(), None
    tracker = PipelineProgress.create(
        total_files=total_files,
        force_progress=show_progress,
        cancel=cancel,
    )
    progress_cb = make_progress_callback(tracker) if tracker.enabled else None
    tracker.start()
    return progress_cb, tracker


def _run_pipeline(
    pipeline: WikiPipeline,
    *,
    progress_cb: ProgressCallback | None,
    output_dir: Path,
    console: Console,
    tracker: PipelineProgress | None,
) -> tuple[CompiledWiki, float]:
    """Run the pipeline, report the elapsed time, and announce the output dir."""
    start_time = time.monotonic()
    wiki = pipeline.run(progress_callback=progress_cb)
    elapsed = time.monotonic() - start_time
    if tracker is None:
        emit_json_done(elapsed, f"Compiled wiki written to {output_dir}")
        json_console = get_err_console()
        json_console.print(f"Compiled wiki written to {output_dir}")
        return wiki, elapsed
    tracker.finish()
    console.print(f"Compiled wiki written to {output_dir}")
    return wiki, elapsed


def _post_compile(
    *,
    wiki: CompiledWiki,
    vp: VaultPaths,
    cfg: ResolvedConfig,
    pipeline_config: PipelineConfig,
    review: ReviewMode,
    bypass_review: BypassReview,
    provenance: Provenance,
    private: Private,
    verify: Verify,
    synthesize: SynthesizeFlag,
    index_store: IndexStore,
    compile_lint: CompileLint,
    compile_lint_all: CompileLintAll,
    embed_model: EmbedModel,
    show_progress: Progress,
    verbose_lint: LintVerbose,
    quiet_lint: LintQuiet,
) -> None:
    """Run every configured after-the-write step (lint, review, index, …)."""
    lint_config = LintConfig(
        fuzzy_threshold=float(cfg.kompend.lint.fuzzy_threshold),
        llm_timeout=float(cfg.kompend.lint.llm_timeout),
    )
    post_compile_actions(
        PostCompileOptions(
            wiki=wiki,
            pipeline_config=pipeline_config,
            output_dir=vp.output_dir,
            sources_dir=vp.sources_dir,
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
            lint_output=resolve_lint_output(cfg.kompend.lint.output, verbose=verbose_lint, quiet=quiet_lint),
        )
    )


def _report_failure(
    exc: Exception,
    *,
    json_progress: bool,
    tracker: PipelineProgress | None,
) -> None:
    """Report a failed compile on the channel the caller listens to.

    A traceback is printed for an unexpected failure only: an ``LLMOutputError``
    is a known verdict with a message worth reading and a stack worth skipping.
    """
    if tracker is not None:
        tracker.stop()
    err_msg = f"Error: {exc}"
    if not isinstance(exc, LLMOutputError):
        get_err_console().print(traceback.format_exc())
    if json_progress:
        err_payload = json.dumps({"event": "error", "message": err_msg}, ensure_ascii=False)
        sys.stdout.write(err_payload + "\n")
        sys.stdout.flush()
    else:
        get_err_console().print(err_msg)
