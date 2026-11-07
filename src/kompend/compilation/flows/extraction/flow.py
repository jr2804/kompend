"""Extraction flow — LLM extraction from source documents.

Extracted from :class:`TwoPhaseCompiler` — extraction phase methods
that operate on compiler instance state.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

from kompend.compilation.asset_mention_rewriter import rewrite_asset_mentions as _rewrite_asset_mentions
from kompend.compilation.asset_registry import AssetIdMap
from kompend.compilation.flows.agentic import extract_agentic
from kompend.compilation.flows.assembly import build_tree_texts
from kompend.compilation.flows.extraction.units import (
    _extract_char_budget,
    _replan_after_overflow,
    _text_groups,
    batch_texts,
    combine_extractions,
    extract_incremental,
    extract_one,
)
from kompend.compilation.two_phase import (
    Extraction,
    ExtractionSet,
    ProgressFn,
    TwoPhaseRuntime,
    rate_suffix,
)
from kompend.core.config._configs import ExtractionStrategy
from kompend.core.models import ChunkTree, ExtractionStats, ResolvedChunk
from kompend.core.paths import rel_to_vault
from kompend.core.templates import get_template_loader
from kompend.core.utils.exceptions import LLMContextOverflowError
from kompend.logging import get_logger

logger = get_logger(__name__)

_loader = get_template_loader()


def run_extraction(  # noqa: PLR0911, PLR0912, PLR0915
    compiler: TwoPhaseRuntime,
    chunks: list[ResolvedChunk],
    *,
    trees: list[ChunkTree] | None = None,
    asset_id_maps: dict[str, AssetIdMap] | None = None,
) -> ExtractionSet:
    """Phase 1: run the LLM in extraction mode for each source.

    When *asset_id_maps* is provided, structured leaves (tables/figures) are
    elided from the extraction text and replaced by ``¶ <id> (summary: …)``
    references — reducing noise and context-window pressure.
    """
    compiler._llm_stats = []
    extractions = ExtractionSet()

    def _rewrite_extraction_mentions(ext: Extraction, id_map: AssetIdMap | None) -> None:
        """Mutate *ext* in place, rewriting figure/table mentions to asset refs."""
        if id_map is None:
            return
        ext.summary = _rewrite_asset_mentions(ext.summary, id_map)
        for c in ext.concepts:
            c.name = _rewrite_asset_mentions(c.name, id_map)
            c.definition = _rewrite_asset_mentions(c.definition, id_map)
        for e in ext.entities:
            e.name = _rewrite_asset_mentions(e.name, id_map)
            e.context = _rewrite_asset_mentions(e.context, id_map)

    by_source: dict[str, list[ResolvedChunk]] = defaultdict(list)
    base_dir = compiler._config.paths.base_dir  # noqa: SLF001
    for rc in chunks:
        label = rel_to_vault(rc.chunk.source_file or "unknown", base_dir)
        by_source[label].append(rc)

    tree_by_source: dict[str, ChunkTree] = {}
    if trees:
        for t in trees:
            if t.source_file is not None:
                tree_by_source[rel_to_vault(t.source_file, base_dir)] = t

    extraction_system = _loader.get_body("compilation/extraction_system.md.j2")
    num_sources = len(by_source)
    logger.info("TwoPhase: extracting %d source(s)", num_sources)

    strategy = compiler._config.llm.extraction_strategy
    agentic = strategy == ExtractionStrategy.AGENTIC

    # The agentic path walks the document section by section; the batch paths
    # walk it batch by batch. The bar reports the same units the flow works in,
    # so a 200-section document shows movement.
    unit = "sections" if agentic else "batches"

    def _source_units(source: str, source_chunks: list[ResolvedChunk]) -> tuple[list[str], list[str]]:
        """The text blocks of one source and the batches they pack into."""
        tree = tree_by_source.get(source)
        id_map = asset_id_maps.get(source) if asset_id_maps else None
        texts = build_tree_texts(tree, asset_mode="ref", asset_id_map=id_map) if tree is not None else [rc.chunk.text for rc in source_chunks if rc.chunk.text]
        # Read the budget per source, not once per run: a provider that
        # lowered the window mid-run must change how the remaining sources
        # are batched.
        return texts, batch_texts(texts, _extract_char_budget(compiler))

    # Pre-compute what each source weighs in the bar from the units the flow
    # sends, so the document's share and its sections' shares come from one
    # measurement (larger sections = more progress). The chunk tier built every
    # tree before this loop, so every count is known before the first call. The
    # texts are rebuilt per source below - holding them all would keep the
    # whole corpus in memory.
    source_unit_totals: list[int] = []
    source_unit_chars: list[int] = []
    for source, source_chunks in by_source.items():
        texts, batches = _source_units(source, source_chunks)
        units = texts if agentic else batches
        source_unit_totals.append(len(units))
        source_unit_chars.append(sum(len(u) for u in units))
    # One-slot lists: a re-plan can change a source's unit count mid-run, and
    # both the corpus-wide total and the bar's weights must follow it.
    total_units_all = [sum(source_unit_totals) or len(by_source)]
    total_chars_all = sum(source_unit_chars) or len(by_source)

    cumulative_chars = 0
    cumulative_units = 0

    for source_idx, (source, source_chunks) in enumerate(by_source.items(), start=1):
        logger.info("TwoPhase: extract [%d/%d] %s", source_idx, num_sources, source)
        src_start = time.monotonic()

        total_chars = source_unit_chars[source_idx - 1]
        chars_before = cumulative_chars
        cumulative_chars += total_chars
        source_name = Path(source).name

        id_map = asset_id_maps.get(source) if asset_id_maps else None
        texts, batches_list = _source_units(source, source_chunks)
        units = texts if agentic else batches_list
        units_before = cumulative_units
        # The pre-pass counted this source before any re-plan; refresh its share
        # of the run-wide total with the count the flow actually works in.
        total_units_all[0] += len(units) - source_unit_totals[source_idx - 1]
        source_unit_totals[source_idx - 1] = len(units)
        cumulative_units += len(units)
        progress = _source_progress(
            compiler,
            source_idx=source_idx,
            num_sources=num_sources,
            source_name=source_name,
            chars_before=chars_before,
            source_chars=total_chars,
            total_chars_all=total_chars_all,
            weights=[len(u) for u in units],
            unit=unit,
            units_before=units_before,
            units_total_all=total_units_all,
        )
        progress(0, len(units), "")

        if not texts:
            empty_ext = Extraction(source=source)
            _rewrite_extraction_mentions(empty_ext, id_map)
            extractions.add(empty_ext)
            compiler._extraction_stats.append(ExtractionStats(source=source))
            progress(len(units), len(units), "")
            continue

        batch_extractions: list[Extraction | None] = []

        if agentic:
            # Agentic: structured memory with compaction — sections are
            # processed sequentially, not batched/merged.
            agentic_result = extract_agentic(compiler, source, texts, extraction_system, progress=progress)
            if agentic_result is not None:
                _rewrite_extraction_mentions(agentic_result, id_map)
                extractions.add(agentic_result)
            compiler._extraction_stats.append(
                ExtractionStats(
                    source=source,
                    summary=agentic_result.summary if agentic_result else "",
                    duration_s=round(time.monotonic() - src_start, 1),
                    summary_chars=len(agentic_result.summary) if agentic_result else 0,
                    summary_words=len(agentic_result.summary.split()) if agentic_result else 0,
                    entity_count=len(agentic_result.entities) if agentic_result else 0,
                    concept_count=len(agentic_result.concepts) if agentic_result else 0,
                    tag_count=len(agentic_result.tags) if agentic_result else 0,
                    batches=len(texts),
                )
            )
            if compiler._batch_tracker.should_stop():
                break
            # Settle the bar at the end of the document — a cache hit
            # returns before the section loop, so no unit was reported.
            progress(len(units), len(units), "")
            continue

        if strategy == ExtractionStrategy.INCREMENTAL and len(batches_list) > 1:
            batch_extractions, _ = extract_incremental(
                compiler,
                source,
                batches_list,
                extraction_system,
                progress=progress,
                texts=texts,
            )
        else:
            batch_extractions = _extract_batches(compiler, source, batches_list, extraction_system, progress, texts=texts)

        valid = [e for e in batch_extractions if e is not None]
        if not valid:
            logger.warning("TwoPhase: all extraction batches failed for '%s' — skipped", source)
        merged = combine_extractions(compiler, source, valid)
        _rewrite_extraction_mentions(merged, id_map)
        extractions.add(merged)
        compiler._extraction_stats.append(
            ExtractionStats(
                source=source,
                summary=merged.summary,
                duration_s=round(time.monotonic() - src_start, 1),
                summary_chars=len(merged.summary),
                summary_words=len(merged.summary.split()),
                entity_count=len(merged.entities),
                concept_count=len(merged.concepts),
                tag_count=len(merged.tags),
                batches=len(batch_extractions),
            )
        )
        if compiler._batch_tracker.should_stop():
            logger.error(
                "TwoPhase: stopping extraction early — %d/%d batches failed (model '%s' is not suitable for structured JSON output)",
                compiler._batch_tracker.failed,
                compiler._batch_tracker.total,
                compiler._model,
            )
            break
    return extractions


def _extract_batches(
    compiler: TwoPhaseRuntime,
    source: str,
    batches_list: list[str],
    extraction_system: str,
    progress: ProgressFn,
    *,
    texts: list[str] | None = None,
) -> list[Extraction | None]:
    """Run the batch extraction path, re-planning when the provider overflows.

    A batch that the provider rejects as too large is not lost work: the
    provider names the limit it enforces, the window is lowered to it, and
    only the texts not yet extracted are re-planned against the smaller
    budget. Completed batches are kept, the failing one is split, and no
    text is dropped or sent twice.
    """
    results: list[Extraction | None] = []
    # The unit of work is a text, not a batch: re-planning must be able to
    # split a batch that grew past the window, and only a text list can.
    pending: list[str] = list(texts) if texts is not None else list(batches_list)
    while pending:
        # Without the unbatched texts there is no text grouping to recover:
        # keep the batches the caller already packed.
        groups = _text_groups(pending, _extract_char_budget(compiler)) if texts is not None else [[batch] for batch in pending]
        extracted = 0
        try:
            for group in groups:
                results.append(extract_one(compiler, source, "\n\n".join(group), extraction_system))
                extracted += len(group)
                progress(len(results), len(groups), "")
                if compiler._batch_tracker.should_stop():
                    return results
            return results
        except LLMContextOverflowError as exc:
            # Only the failing batch and the later ones are left: the results
            # above are kept and never sent again.
            pending = _replan_after_overflow(compiler, source, pending[extracted:], exc)
    return results


def _source_progress(
    compiler: TwoPhaseRuntime,
    *,
    source_idx: int,
    num_sources: int,
    source_name: str,
    chars_before: int,
    source_chars: int,
    total_chars_all: int,
    weights: list[int],
    unit: str,
    units_before: int,
    units_total_all: list[int],
) -> ProgressFn:
    """A progress callback for the units of one source.

    The bar counts characters, so a document's share is paid out as its
    units complete: the bar moves *inside* a long document instead of
    jumping when the document starts. A source without units (nothing to
    extract) is paid out in full at once. The line carries the corpus-wide
    unit counter as well, so the position in the whole run is readable in
    the unit the flow works in.

    *units_total_all* is the run-wide counter (a one-slot list, mutated in
    place): a re-plan changes how many units a source works in, and the
    counter must follow it instead of reporting a total the bar can never
    reach.
    """
    weight_total = sum(weights)
    # The unit count the per-unit *weights* describe. A stage that re-plans
    # reports a different total; the weights are then stale and the document's
    # share is paid evenly across the units it actually works in.
    known_total = [len(weights)]

    def report(units_done: int, units_total: int, activity: str) -> None:
        if units_total != known_total[0]:
            units_total_all[0] += units_total - known_total[0]
            known_total[0] = units_total
        if not weight_total:
            done, fraction = units_done, 1.0
        elif units_total == len(weights):
            done = max(0, min(units_done, len(weights)))
            fraction = sum(weights[:done]) / weight_total
        else:
            done = max(0, min(units_done, units_total))
            fraction = done / units_total if units_total else 1.0
        chars = chars_before + int(source_chars * fraction)
        detail = extraction_detail(
            source_idx,
            num_sources,
            source_name,
            done,
            units_total,
            unit,
            activity,
            units_before=units_before,
            units_total_all=units_total_all[0],
        )
        compiler._report(chars, total_chars_all, detail + rate_suffix(compiler._usage_tracker))

    return report


def extraction_detail(
    source_idx: int,
    num_sources: int,
    source_name: str,
    units_done: int,
    units_total: int,
    unit: str,
    activity: str = "",
    *,
    units_before: int = 0,
    units_total_all: int = 0,
) -> str:
    """The extraction status line for the progress bar.

    Names the document, how many of its units are done, and — while a
    single slow call runs — what that call is doing. The corpus-wide
    counter rides in the same parenthesis, in the unit just named::

        extracting [3/100] PolqaOemLibManualV3.18.pdf (12/234 · 2450/9800 sections), consolidating context
    """
    counter = f"{units_done}/{units_total}"
    if units_total_all:
        counter += f" · {units_before + units_done}/{units_total_all}"
    detail = f"extracting [{source_idx}/{num_sources}] {source_name} ({counter} {unit})"
    if activity:
        detail += f", {activity}"
    return detail
