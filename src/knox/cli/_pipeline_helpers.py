"""Internal helpers for pipeline CLI commands.

Private API — external callers should use ``knox.cli.cmds.pipeline``.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from knox.cli._progress import PipelineProgress
from knox.cli._sources import download_urls, ensure_in_sources, resolve_files
from knox.cli.args import (
    CompileFiles,
    Ignore,
    Recursive,
    UrlFile,
)
from knox.core.models import CompiledWiki, MetadataField, WikiField
from knox.core.paths import LINT_HISTORY_FILE, VaultPaths
from knox.core.pipeline_defs import PipelineConfig
from knox.lint.history import entry_from_report, save_lint_entry
from knox.lint.runner import LintRunner, report_to_summary
from knox.logging import get_console, get_err_console, get_logger
from knox.output.wiki_index import write_index, write_overview
from knox.provenance import annotate_wiki_with_provenance
from knox.rag import build_vector_index
from knox.review import ReviewQueue
from knox.search.index import build_chunk_index
from knox.trust import mark_wiki_private
from knox.verification import verify_wiki

logger = get_logger(__name__)


def _make_progress_callback(tracker: PipelineProgress) -> Callable[[str, int, int, str], None]:  # noqa: PLR0915
    """Build a progress callback that translates pipeline events into Rich updates.

    The overall bar resets per phase. Each phase has a flat handler keyed by
    name in the dispatch table — no nested phase if/elif chain.
    - **parse**: bar tracks files (0 → N)
    - **resolve**: bar resets to track chunks (0 → M)
    - **compile** / **output**: bar becomes indeterminate
    """

    def _parse(current: int, total: int, detail: str) -> None:
        if current == 0:
            tracker.start()
        elif detail == "done":
            tracker.update_progress(total, f"Parsed {total} file(s)")
        else:
            tracker.update_progress(current, f"[{current}/{total}] {detail}")

    def _index(current: int, total: int, detail: str) -> None:
        if current == 0:
            tracker.start()
            tracker.message("Downloading sources…")
        else:
            tracker.message("Download complete")

    def _resolve(current: int, total: int, detail: str) -> None:
        if current == 0:
            if detail == "vlm":
                tracker.reset_for_phase(total, f"Captioning images ({total})…")
            else:
                tracker.reset_for_phase(total, f"Resolving chunks ({total} total)…")
        elif detail == "done":
            tracker.update_progress(total, "Resolve complete")
        elif detail == "captioning":
            tracker.update_progress(current, f"Captioning image [{current}/{total}]")
        else:
            tracker.update_progress(current, f"Resolving [{current}/{total}]")

    def _start_done(start_msg: str, done_msg: str) -> Callable[[int, int, str], None]:
        """output: indeterminate bar driven by start/done events.

        Events arrive as ``_emit(phase, current, total, detail)``:
        - ``detail == "starting"`` → reset to indeterminate bar
        - ``detail == "done"``      → mark phase complete
        """

        def _cb(current: int, total: int, detail: str) -> None:
            if detail == "done":
                tracker.message(done_msg)
            elif detail == "starting":
                tracker.reset_for_phase(None, start_msg)

        return _cb

    def _compile() -> Callable[[int, int, str], None]:
        """compile: DETERMINATE bar counting real work units.

        The compiler reports ``(current, total, description)`` where
        *total* is the number of countable units (e.g. source extractions).
        The bar fills as work completes instead of faking activity. The
        bar resets once when *total* is first known, then advances on each
        report and fills to 100% on "done".
        """
        state: dict[str, int | None] = {"total": None}

        def _cb(current: int, total: int, detail: str) -> None:
            if detail == "done":
                if state["total"] is not None:
                    tracker.update_progress(state["total"], "Compile complete")
                else:
                    tracker.message("Compile complete")
            elif detail == "starting":
                tracker.reset_for_phase(None, "Compiling wiki…")
                state["total"] = None
            elif total and total != state["total"]:
                state["total"] = total
                tracker.reset_for_phase(total, "Compiling wiki…")
                tracker.update_progress(current, detail)
            else:
                tracker.update_progress(current, detail)

        return _cb

    handlers: dict[str, Callable[[int, int, str], None]] = {
        "parse": _parse,
        "index": _index,
        "resolve": _resolve,
        "compile": _compile(),
        "output": _start_done("Writing output…", "Output complete"),
    }

    def _cb(phase: str, current: int, total: int, detail: str) -> None:
        handler = handlers.get(phase)
        if handler is not None:
            handler(current, total, detail)

    return _cb


def _resolve_compile_inputs(
    files: CompileFiles,
    online_urls: list[str],
    url_file: UrlFile,
    sources_dir: Path,
    recursive: Recursive,
    ignore: Ignore,
    *,
    in_place: bool = False,
    undetected: bool = True,
) -> list[Path]:
    """Resolve file arguments and URLs into a deduplicated input file list.

    When *in_place* is ``True``, local files outside *sources_dir* are
    kept at their original location instead of being copied.
    """
    if url_file is not None:
        for raw_line in url_file.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#"):
                online_urls.append(stripped)

    resolved = resolve_files(*files, recursive=recursive, ignore_patterns=ignore)

    sources_dir.mkdir(parents=True, exist_ok=True)
    local_files = ensure_in_sources(resolved, sources_dir, in_place=in_place)
    downloaded = download_urls(online_urls, sources_dir, base_dir=sources_dir.parent, undetected=undetected)

    all_input_files = list(dict.fromkeys(local_files + downloaded))
    if not all_input_files:
        all_input_files = resolve_files(
            str(sources_dir),
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
    paths = VaultPaths(base_dir)
    cache_file = paths.wiki_cache_path
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    def _rel_path(p: str) -> str:
        try:
            return str(Path(p).relative_to(base_dir))
        except ValueError:
            return p

    nodes_dicts: list[dict[str, Any]] = []
    for n in wiki.nodes:
        nd = dataclasses.asdict(n)
        meta = nd.get("metadata", {})
        if MetadataField.SOURCE_FILE in meta:
            meta = dict(meta)
            meta[MetadataField.SOURCE_FILE] = _rel_path(meta[MetadataField.SOURCE_FILE])
            nd["metadata"] = meta
        nodes_dicts.append(nd)

    data = {
        WikiField.NODES: nodes_dicts,
        WikiField.EDGES: list(wiki.edges),
        WikiField.TOC: list(wiki.toc),
        WikiField.QUALITY_SCORE: wiki.quality_score,
        WikiField.ITERATION: wiki.iteration,
        "metadata": dict(wiki.metadata),
    }
    cache_file.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")


def _print_tier_listing(pipeline: object) -> None:
    """Print the tier listing from a WikiPipeline."""
    if not hasattr(pipeline, "list_tiers"):
        get_err_console().print("Pipeline does not support tier listing.")
        return

    get_console().print(json.dumps(pipeline.list_tiers(), indent=2, default=str))  # type: ignore[union-attr]  # ty:ignore[call-non-callable]


def _send_to_review_queue(wiki: CompiledWiki, base_dir: Path) -> None:
    """Send compiled wiki nodes to the review queue."""
    queue = ReviewQueue(review_dir=VaultPaths(base_dir).review_dir)
    for node in wiki.nodes:
        queue.enqueue(node)


def _annotate_provenance(wiki: CompiledWiki, output_dir: Path) -> None:
    """Annotate compiled wiki with claim-level provenance citations."""
    annotated = annotate_wiki_with_provenance(wiki, output_dir)
    get_console().print(f"Annotated {len(annotated.nodes)} nodes with provenance")


def _mark_private(wiki: CompiledWiki, output_dir: Path) -> None:
    """Mark all wiki pages as private via frontmatter."""
    mark_wiki_private(wiki, output_dir)


def _verify_claims(
    wiki: CompiledWiki,
    output_dir: Path,
    *,
    client: object,
    model: str,
    verify_budget: int = 3000,
) -> None:
    """Run grounding verification over wiki claims."""
    result = verify_wiki(wiki, output_dir, client=client, model=model, verify_budget=verify_budget)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    get_console().print(f"Verified {result.verified}/{result.total} claims (avg confidence {result.average_confidence:.2f})")


def _run_post_compile_lint(
    wiki: CompiledWiki,
    sources_dir: Path,
    *,
    base_dir: Path,
    json_output: bool = False,
) -> None:
    """Run lint after compilation, persist to history, and print a summary."""
    runner = LintRunner(sources_dir=str(sources_dir) if sources_dir.exists() else None)
    lint_start = time.monotonic()
    report = runner.run(wiki)
    lint_elapsed = time.monotonic() - lint_start

    state_dir = VaultPaths(base_dir).knox_dir
    entry = entry_from_report(
        report,
        duration_seconds=lint_elapsed,
        options={"context": "pipeline"},
        config=type("Cfg", (), {"get": staticmethod(lambda k: None)})(),
        node_count=len(wiki.nodes),
    )
    save_lint_entry(entry, state_dir / LINT_HISTORY_FILE)

    get_console().print(f"\n{report_to_summary(report)}")
    if report.failures > 0:
        get_console().print(
            f"  Details: `knox lint run` (full report) or `knox lint gaps` (source recommendations)\n"
            f"  History: `knox lint history` — saved to .knox/{LINT_HISTORY_FILE}"
        )


def _cli_build_vector_index(
    wiki: CompiledWiki,
    output_dir: Path,
    embed_model: str | None,
    *,
    base_dir: Path | None = None,
) -> None:
    """Thin CLI wrapper — delegates to knox.rag and prints status."""
    if not embed_model:
        get_err_console().print("Error: --embed-model must be provided when --index is used.")
        raise typer.Exit(1)

    count = build_vector_index(wiki, output_dir, embed_model=embed_model, base_dir=base_dir)
    if count > 0:
        get_console().print(f"Indexed {count} nodes in zvec store for semantic search")
    else:
        get_err_console().print("Warning: no nodes indexed — vector index skipped")


def _print_compile_summary(wiki: CompiledWiki) -> None:
    """Print a one-line compilation summary with timing and LLM usage."""
    meta = wiki.metadata
    parts = [f"{len(wiki.nodes)} page(s)"]

    chunk_count = meta.get(WikiField.CHUNK_COUNT, 0)
    if chunk_count:
        parts.append(f"{chunk_count} chunk(s)")

    dur_s = meta.get("compile_duration_s")
    if dur_s is not None:
        parts.append(f"{dur_s:.0f}s")

    src_stats = meta.get("extraction_stats")
    if src_stats:
        parts.append(f"{len(src_stats)} source(s)")

    llm_calls = meta.get("llm_calls")
    if llm_calls:
        pt = meta.get("llm_prompt_tokens", 0)
        ct = meta.get("llm_completion_tokens", 0)
        parts.append(f"{llm_calls} LLM call(s) ({pt}+{ct}={pt + ct} token(s))")

    wicer_iters = meta.get("wicer_iterations")
    if wicer_iters:
        wicer_refs = meta.get("wicer_refinements", 0)
        parts.append(f"WiCER {wicer_iters} iter(s), {wicer_refs} refinement(s)")
        wc = meta.get("wicer_llm_calls")
        if wc:
            parts.append(f"+{wc} probe/eval call(s)")

    get_console().print(f"Compiled: {', '.join(parts)}")

    if meta.get("generation_skipped"):
        reason = meta.get("skip_reason", "unknown")
        concepts = meta.get("concepts_available", 0)
        get_err_console().print(
            f"Warning: generation skipped — 0 pages written ({reason}). "
            f"{concepts} concept(s) were extracted but could not be synthesized. "
            f"Check LLM backend and re-run."
        )


def _run_overview_synthesis(
    *,
    wiki: CompiledWiki,
    pipeline_config: PipelineConfig,
    base_dir: Path,
    show_progress: bool | None,
) -> None:
    """Generate overview.md via LLM synthesis."""
    resolved_model = pipeline_config.llm.default_model
    synthesize_tracker = PipelineProgress.create(total_files=1, force_progress=show_progress)
    synthesize_tracker.start()

    def _synth_cb(current: int, total: int) -> None:
        if current == 0:
            synthesize_tracker.reset_for_phase(total, "Synthesizing overview…")
        else:
            synthesize_tracker.update_progress(current, f"[{current}/{total}] Synthesizing overview…")

    try:
        overview_path = write_overview(
            wiki,
            base_dir,
            model=resolved_model,
            overview_budget=pipeline_config.budget.overview_output_max * 4,
            timeout=pipeline_config.llm.timeout,
            progress_callback=_synth_cb if synthesize_tracker.enabled else None,
        )
    except Exception as exc:
        synthesize_tracker.finish()
        get_err_console().print(f"Overview generation failed: {exc}")
        return
    synthesize_tracker.finish()
    if overview_path:
        get_console().print(f"Overview: {overview_path}")
    else:
        get_err_console().print("Overview generation skipped (no content to summarise).")


def _post_compile_actions(
    *,
    wiki: CompiledWiki,
    pipeline_config: PipelineConfig,
    output_dir: Path,
    sources_dir: Path,
    review: bool | None,
    bypass_review: bool | None,
    provenance: bool | None,
    private: bool | None,
    verify: bool | None,
    synthesize: bool | None,
    index_store: bool | None,
    compile_lint: bool | None,
    embed_model: str | None,
    show_progress: bool | None = None,
    llm_client: object | None = None,
) -> None:
    """Run all post-compilation steps (review, provenance, index, lint, etc.)."""
    base_dir = pipeline_config.resolved_base_dir()
    _write_wiki_cache(wiki, base_dir)
    _print_compile_summary(wiki)

    if review and not bypass_review:
        _send_to_review_queue(wiki, base_dir)
        get_console().print(f"Pages pending review in {VaultPaths(base_dir).review_dir}")
    elif bypass_review and review:
        get_console().print("--bypass-review overrides --review; writing pages directly.")

    if provenance:
        _annotate_provenance(wiki, output_dir)

    if private:
        _mark_private(wiki, output_dir)

    if verify:
        resolved_model = pipeline_config.llm.default_model
        if llm_client is None:
            get_err_console().print("Warning: --verify requires an LLM client — skipping verification")
        else:
            _verify_claims(wiki, output_dir, client=llm_client, model=resolved_model, verify_budget=pipeline_config.budget.verify_output_max * 4)

    generated = write_index(wiki, output_dir, base_dir=base_dir)
    get_console().print(f"Index: {generated['index']}")

    if synthesize:
        _run_overview_synthesis(
            wiki=wiki,
            pipeline_config=pipeline_config,
            base_dir=base_dir,
            show_progress=show_progress,
        )

    if index_store:
        search_db_path = VaultPaths(base_dir).search_db_path
        build_chunk_index(wiki, search_db_path)
        wiki.metadata["embedding_model"] = embed_model
        _cli_build_vector_index(wiki, output_dir, pipeline_config.embedding.model, base_dir=base_dir)

    if compile_lint:
        _run_post_compile_lint(wiki, sources_dir, base_dir=base_dir)

    _write_wiki_cache(wiki, base_dir)
