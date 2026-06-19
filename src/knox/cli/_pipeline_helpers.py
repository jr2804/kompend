"""Internal helpers for pipeline CLI commands.

Private API — external callers should use ``knox.cli.pipeline_cmds``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from pathlib import Path

import typer

from knox.cli._progress import PipelineProgress
from knox.cli._sources import download_urls, ensure_in_sources, resolve_files
from knox.cli.args import (
    CompileFiles,
    Ignore,
    Recursive,
    UrlFile,
)
from knox.core.models import CompiledWiki
from knox.core.paths import VaultPaths
from knox.core.pipeline_defs import PipelineConfig
from knox.lint.runner import LintRunner, report_to_text
from knox.output.wiki_index import write_index_and_log, write_overview
from knox.provenance import annotate_wiki_with_provenance
from knox.rag import build_vector_index
from knox.review import ReviewQueue
from knox.search.index import build_chunk_index
from knox.trust import mark_wiki_private
from knox.verification import verify_wiki

logger = logging.getLogger(__name__)


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
    *,
    in_place: bool = False,
) -> list[Path]:
    """Resolve file arguments and URLs into a deduplicated input file list.

    When *in_place* is ``True``, local files outside *sources_dir_path* are
    kept at their original location instead of being copied.
    """
    if url_file is not None:
        for raw_line in url_file.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#"):
                online_urls.append(stripped)

    resolved = resolve_files(*files, recursive=recursive, ignore_patterns=ignore)

    sources_dir_path.mkdir(parents=True, exist_ok=True)
    local_files = ensure_in_sources(resolved, sources_dir_path, in_place=in_place)
    downloaded = download_urls(online_urls, sources_dir_path, base_dir=sources_dir_path.parent)

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
    paths = VaultPaths(base_dir)
    cache_file = paths.wiki_cache_path
    cache_file.parent.mkdir(parents=True, exist_ok=True)

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

    typer.echo(json.dumps(pipeline.list_tiers(), indent=2, default=str))  # type: ignore[union-attr]  # ty:ignore[call-non-callable]


def _send_to_review_queue(wiki: CompiledWiki, base_dir: Path) -> None:
    """Send compiled wiki nodes to the review queue."""
    queue = ReviewQueue(review_dir=VaultPaths(base_dir).review_dir)
    for node in wiki.nodes:
        queue.enqueue(node)


def _annotate_provenance(wiki: CompiledWiki, output_dir_path: Path) -> None:
    """Annotate compiled wiki with claim-level provenance citations."""
    annotated = annotate_wiki_with_provenance(wiki, output_dir_path)
    typer.echo(f"Annotated {len(annotated.nodes)} nodes with provenance")


def _mark_private(wiki: CompiledWiki, output_dir_path: Path) -> None:
    """Mark all wiki pages as private via frontmatter."""
    mark_wiki_private(wiki, output_dir_path)


def _verify_claims(
    wiki: CompiledWiki,
    output_dir_path: Path,
    *,
    client: object,
    model: str,
    verify_budget: int = 3000,
) -> None:
    """Run grounding verification over wiki claims."""
    result = verify_wiki(wiki, output_dir_path, client=client, model=model, verify_budget=verify_budget)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    typer.echo(f"Verified {result.verified}/{result.total} claims (avg confidence {result.average_confidence:.2f})")


def _run_post_compile_lint(
    wiki: CompiledWiki,
    sources_dir_path: Path,
    *,
    json_output: bool = False,
) -> None:
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
    show_progress: bool | None = None,
    llm_client: object | None = None,
) -> None:
    """Run all post-compilation steps (review, provenance, index, lint, etc.)."""
    base_dir = pipeline_config.resolved_base_dir()
    _write_wiki_cache(wiki, base_dir)

    if review and not bypass_review:
        _send_to_review_queue(wiki, base_dir)
        typer.echo(f"Pages pending review in {VaultPaths(base_dir).review_dir}")
    elif bypass_review and review:
        typer.echo("--bypass-review overrides --review; writing pages directly.")

    if provenance:
        _annotate_provenance(wiki, output_dir_path)

    if private:
        _mark_private(wiki, output_dir_path)

    if verify:
        resolved_model = pipeline_config.llm.default_model
        if llm_client is None:
            typer.echo("Warning: --verify requires an LLM client — skipping verification", err=True)
        else:
            _verify_claims(wiki, output_dir_path, client=llm_client, model=resolved_model, verify_budget=pipeline_config.budget.verify_output_max * 4)

    generated = write_index_and_log(wiki, output_dir_path, base_dir=base_dir)
    typer.echo(f"Index: {generated['index']}")
    typer.echo(f"Log:   {generated['log']}")

    if synthesize:
        resolved_model = pipeline_config.llm.default_model

        synthesize_tracker = PipelineProgress.create(total_files=1, force_progress=show_progress)
        synthesize_tracker.start()

        def _synth_cb(current: int, total: int) -> None:
            if current == 0:
                synthesize_tracker.reset_for_phase(total, "Synthesizing overview…")
            else:
                synthesize_tracker.update_progress(current, f"[{current}/{total}] Synthesizing overview…")

        overview_path = write_overview(
            wiki,
            base_dir,
            model=resolved_model,
            overview_budget=pipeline_config.budget.overview_output_max * 4,
            timeout=pipeline_config.llm.timeout,
            progress_callback=_synth_cb if synthesize_tracker.enabled else None,
        )
        synthesize_tracker.finish()
        if overview_path:
            typer.echo(f"Overview: {overview_path}")
        else:
            typer.echo("Overview generation skipped (LLM client unavailable).", err=True)

    if index_store:
        search_db_path = VaultPaths(base_dir).search_db_path
        build_chunk_index(wiki, search_db_path)
        wiki.metadata["embedding_model"] = embed_model
        _cli_build_vector_index(wiki, output_dir_path, pipeline_config.embedding.model, base_dir=base_dir)

    if compile_lint:
        _run_post_compile_lint(wiki, sources_dir_path, json_output=False)

    _write_wiki_cache(wiki, base_dir)
