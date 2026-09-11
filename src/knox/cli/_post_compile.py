"""Post-compilation actions — review, provenance, lint, vector index, overview.

Extracted from :mod:`knox.cli._pipeline_helpers` to reduce module size.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

import typer

from knox import __version__ as _KNOX_VERSION
from knox.agents.verification import AgenticVerificationAgent, VerificationAgent, make_agentic_verification_agent
from knox.agents.wiki_index import make_agentic_overview_agent
from knox.cli._progress import PipelineProgress
from knox.core.cache.wiki import write_wiki_cache  # noqa: F401 — re-export
from knox.core.config import resolve_config  # noqa: PLC0415
from knox.core.config._configs import LintConfig
from knox.core.models import CompiledWiki, MetadataField, WikiField
from knox.core.paths import LINT_HISTORY_FILE, VaultPaths
from knox.core.pipeline.config import PipelineConfig
from knox.core.utils.time import iso_local
from knox.lint.flow import run_full_lint_flow, run_runner_with_progress  # noqa: PLC0415
from knox.lint.history import entry_from_report, save_lint_entry
from knox.lint.runner import LintRunner, report_to_summary
from knox.logging import get_console, get_err_console, get_logger
from knox.output.wiki_index import write_index, write_overview
from knox.provenance import annotate_wiki_with_provenance
from knox.rag import build_vector_index
from knox.review import ReviewQueue
from knox.trust import mark_wiki_private
from knox.verification import verify_wiki

# OKF v0.2 (§5.2/§7) trust actor identifying the producer of a verification pass.
_VERIFIED_BY = f"knox/{_KNOX_VERSION or '0.0.0'}"

logger = get_logger(__name__)


@dataclasses.dataclass
class PostCompileOptions:
    """Options for post-compilation actions, grouped to reduce parameter count."""

    wiki: CompiledWiki
    pipeline_config: PipelineConfig
    output_dir: Path
    sources_dir: Path
    review: bool | None = None
    bypass_review: bool | None = None
    provenance: bool | None = None
    private: bool | None = None
    verify: bool | None = None
    synthesize: bool | None = None
    index_store: bool | None = None
    compile_lint: bool | None = None
    compile_lint_all: bool | None = None
    embed_model: str | None = None
    show_progress: bool | None = None
    lint_config: LintConfig | None = None
    embed_attachments: bool = False


def post_compile_actions(opts: PostCompileOptions) -> None:
    """Run all post-compilation steps (review, provenance, index, lint, etc.)."""
    base_dir = opts.pipeline_config.resolved_base_dir()
    write_wiki_cache(opts.wiki, base_dir)
    print_compile_summary(opts.wiki)

    if opts.review and not opts.bypass_review:
        send_to_review_queue(opts.wiki, base_dir)
        get_console().print(f"Pages pending review in {VaultPaths(base_dir).review_dir}")
    elif opts.bypass_review and opts.review:
        get_console().print("--bypass-review overrides --review; writing pages directly.")

    if opts.provenance:
        annotate_provenance(opts.wiki, base_dir)

    if opts.private:
        mark_private(opts.wiki, base_dir)

    if opts.verify:
        model = opts.pipeline_config.llm.default_model
        if not model:
            get_err_console().print("Warning: --verify requires an LLM model — skipping verification")
        else:
            verify_claims(
                opts.wiki,
                base_dir,
                agent=make_agentic_verification_agent(
                    model_string=model,
                    timeout=opts.pipeline_config.llm.timeout,
                    retries=opts.pipeline_config.llm.output_retries,
                    output_retries=opts.pipeline_config.llm.agentic_output_retries if opts.pipeline_config.llm.output_retries_enabled else None,
                ),
                verify_budget=opts.pipeline_config.budget.verify_output_max * 4,
            )

    generated = write_index(opts.wiki, opts.output_dir, base_dir=base_dir)
    get_console().print(f"Index: {generated['index']}")

    if opts.synthesize:
        run_overview_synthesis(
            wiki=opts.wiki,
            pipeline_config=opts.pipeline_config,
            base_dir=base_dir,
            show_progress=opts.show_progress,
        )

    # Build vector index unless explicitly disabled (--no-index) or no
    # embedding model is configured (embedding is opt-in, no default model).
    embed_model = opts.embed_model or opts.pipeline_config.embedding.model
    if opts.index_store is not False and embed_model:
        opts.wiki.metadata[str(MetadataField.EMBEDDING_MODEL)] = embed_model
        try:
            cli_build_vector_index(
                opts.wiki,
                opts.output_dir,
                embed_model,
                base_dir=base_dir,
                embed_attachments=opts.embed_attachments,
            )
        except Exception as exc:
            logger.warning("Vector index build failed (wiki is already compiled). Run 'knox index' to retry: %s", exc)

    if opts.compile_lint_all:
        run_post_compile_lint_all(opts.wiki, base_dir, pipeline_config=opts.pipeline_config, show_progress=opts.show_progress)
    elif opts.compile_lint:
        run_post_compile_lint(opts.wiki, opts.sources_dir, base_dir=base_dir, lint_config=opts.lint_config, show_progress=opts.show_progress)

    write_wiki_cache(opts.wiki, base_dir)


def send_to_review_queue(wiki: CompiledWiki, base_dir: Path) -> None:
    """Send compiled wiki nodes to the review queue."""
    queue = ReviewQueue(review_dir=VaultPaths(base_dir).review_dir)
    for node in wiki.nodes:
        queue.enqueue(node)


def annotate_provenance(wiki: CompiledWiki, base_dir: Path) -> None:
    """Annotate compiled wiki with claim-level provenance citations."""
    annotated = annotate_wiki_with_provenance(wiki, base_dir)
    get_console().print(f"Annotated {len(annotated.nodes)} nodes with provenance")


def mark_private(wiki: CompiledWiki, base_dir: Path) -> None:
    """Mark all wiki pages as private via frontmatter."""
    mark_wiki_private(wiki, base_dir)


def verify_claims(
    wiki: CompiledWiki,
    base_dir: Path,
    *,
    agent: VerificationAgent | AgenticVerificationAgent,
    verify_budget: int = 3000,
) -> None:
    """Run grounding verification over wiki claims.

    Records the verification pass as an OKF v0.2 ``verified`` trust event
    (§5.2) on ``wiki.metadata`` so output writers can surface it. The actor
    follows the §7 convention (``<producer>/<version>``).
    """
    result = verify_wiki(wiki, base_dir, agent=agent, verify_budget=verify_budget)
    get_console().print(f"Verified {result.verified}/{result.total} claims (avg confidence {result.average_confidence:.2f})")

    # Stamp the OKF v0.2 ``verified`` trust event (§5.2).
    verified_events = list(wiki.metadata.get("verified") or [])
    verified_events.append({"by": _VERIFIED_BY, "at": iso_local()})  # TrustEvent (output.models)
    wiki.metadata["verified"] = verified_events


def run_post_compile_lint_all(
    wiki: CompiledWiki,
    base_dir: Path,
    *,
    pipeline_config: PipelineConfig,
    show_progress: bool | None = None,
) -> dict[str, Any]:

    paths = VaultPaths(base_dir)
    cfg = resolve_config(base_dir=base_dir)
    return run_full_lint_flow(
        paths,
        cfg,
        fix=True,
        json_output=False,
        show_progress=show_progress if show_progress is not None else True,
    )


def run_post_compile_lint(
    wiki: CompiledWiki,
    sources_dir: Path,
    *,
    base_dir: Path,
    json_output: bool = False,
    lint_config: LintConfig | None = None,
    show_progress: bool | None = None,
) -> None:
    """Run lint after compilation, persist to history, and print a summary."""
    runner = LintRunner(
        sources_dir=str(sources_dir) if sources_dir.exists() else None,
        lint_config=lint_config,
    )
    lint_start = time.monotonic()
    report = run_runner_with_progress(runner, wiki, show_progress=show_progress)
    lint_elapsed = time.monotonic() - lint_start

    state_dir = VaultPaths(base_dir).knox_dir
    entry = entry_from_report(
        report,
        duration_seconds=lint_elapsed,
        options={"context": "pipeline"},
        config=dict[str, Any](),
        node_count=len(wiki.nodes),
    )
    save_lint_entry(entry, state_dir / LINT_HISTORY_FILE)

    get_console().print(f"\n{report_to_summary(report)}")
    if report.failures > 0:
        get_console().print(
            f"  Details: `knox lint run` (full report) or `knox lint gaps` (source recommendations)\n"
            f"  History: `knox lint history` — saved to .knox/{LINT_HISTORY_FILE}"
        )


def cli_build_vector_index(
    wiki: CompiledWiki,
    output_dir: Path,
    embed_model: str | None,
    *,
    base_dir: Path | None = None,
    embed_attachments: bool = False,
) -> None:
    """Thin CLI wrapper — delegates to knox.rag and prints status."""
    if not embed_model:
        get_err_console().print("Error: --embed-model must be provided when --index is used.")
        raise typer.Exit(1)

    count = build_vector_index(
        wiki,
        output_dir,
        embed_model=embed_model,
        base_dir=base_dir,
        embed_attachments=embed_attachments,
    )
    if count > 0:
        get_console().print(f"Indexed {count} nodes in zvec store for semantic search")
    else:
        get_err_console().print("Warning: no nodes indexed — vector index skipped")


def print_compile_summary(wiki: CompiledWiki) -> None:
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

    if meta.get(MetadataField.GENERATION_SKIPPED):
        reason = meta.get(MetadataField.SKIP_REASON, "unknown")
        concepts = meta.get(MetadataField.CONCEPTS_AVAILABLE, 0)
        get_err_console().print(
            f"Warning: generation skipped — 0 pages written ({reason}). "
            f"{concepts} concept(s) were extracted but could not be synthesized. "
            f"Check LLM backend and re-run."
        )


def run_overview_synthesis(
    *,
    wiki: CompiledWiki,
    pipeline_config: PipelineConfig,
    base_dir: Path,
    show_progress: bool | None,
) -> None:
    """Generate overview.md via LLM synthesis."""
    synthesize_tracker = PipelineProgress.create(total_files=1, force_progress=show_progress)
    synthesize_tracker.start()

    def _synth_cb(current: int, total: int) -> None:
        if current == 0:
            synthesize_tracker.reset_for_phase(total, "Synthesizing overview…")
        else:
            synthesize_tracker.update_progress(current, f"[{current}/{total}] Synthesizing overview…")

    try:
        overview_file = write_overview(
            wiki,
            base_dir,
            agent=make_agentic_overview_agent(
                model_string=pipeline_config.llm.default_model or "",
                timeout=pipeline_config.llm.timeout,
                retries=pipeline_config.llm.output_retries if pipeline_config.llm.output_retries_enabled else 0,
                output_retries=pipeline_config.llm.agentic_output_retries if pipeline_config.llm.output_retries_enabled else 0,
            ),
            overview_budget=pipeline_config.budget.overview_output_max * 4,
            progress_callback=_synth_cb if synthesize_tracker.enabled else None,
        )
    except Exception as exc:
        synthesize_tracker.finish()
        get_err_console().print(f"Overview generation failed: {exc}")
        return
    synthesize_tracker.finish()
    if overview_file:
        get_console().print("Overview: {overview_file}")
    else:
        get_err_console().print("Overview generation skipped (no content to summarise).")
