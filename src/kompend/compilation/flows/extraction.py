"""Extraction flow — LLM extraction from source documents.

Extracted from :class:`TwoPhaseCompiler` — extraction phase methods
that operate on compiler instance state.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.extraction import ExtractionAgent
from kompend.agents.extraction_reduce import ExtractionReduceAgent
from kompend.compilation._two_phase_core import (
    Extraction,
    ExtractionSet,
    ProgressFn,
    TwoPhaseRuntime,
    merge_extractions,
    rate_suffix,
    try_pydanticai_extraction,
)
from kompend.compilation.asset_mention_rewriter import rewrite_asset_mentions as _rewrite_asset_mentions
from kompend.compilation.asset_registry import AssetIdMap
from kompend.compilation.flows.agentic import extract_agentic
from kompend.compilation.flows.assembly import build_tree_texts
from kompend.compilation.schema import ExtractionOutput
from kompend.core.config._configs import ExtractionStrategy
from kompend.core.models import ChunkTree, ExtractionStats, ResolvedChunk
from kompend.core.models.enums import CacheDeps
from kompend.core.paths import rel_to_vault
from kompend.core.templates import get_template_loader
from kompend.core.text.windows import split_text_windows
from kompend.core.utils.exceptions import LLMContextOverflowError, LLMOutputError
from kompend.llm.format import encode_payload
from kompend.llm.request_budget import RequestBudget, chars_to_tokens
from kompend.logging import LOG_SAMPLE_LIMIT, get_logger

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


def batch_texts(texts: list[str], char_budget: int) -> list[str]:
    """Group texts into batches that fit within *char_budget* (chars)."""
    return ["\n\n".join(group) for group in _text_groups(texts, char_budget)]


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


def extract_one(compiler: TwoPhaseRuntime, source: str, combined: str, extraction_system: str) -> Extraction | None:
    """Run a single extraction LLM call with cache."""
    extract_deps = _extract_cache_deps(compiler, combined)
    if not compiler._config.no_cache:
        cached = compiler._cache_store.get_dict("extract", extract_deps)
        if cached is not None:
            logger.info("TwoPhase: extraction cache hit for '%s'", source)
            return Extraction.from_dict(cached)
        logger.info(
            "TwoPhase: extraction cache miss for '%s' (text_hash=%.16s, chunk_min=%s, chunk_max=%s)",
            source,
            extract_deps[CacheDeps.SOURCE_TEXT],
            extract_deps[CacheDeps.CHUNK_SIZE_MIN],
            extract_deps[CacheDeps.CHUNK_SIZE_MAX],
        )

    agent = ExtractionAgent(
        model_string=compiler._model or "",
        timeout=compiler._timeout,
        skills=compiler._config.skills,
        retries=compiler._config.llm.output_retries if compiler._config.llm.output_retries_enabled else 0,
        delta_retry_enabled=compiler._config.llm.delta_retry_enabled,
        reasoning_effort=compiler._config.llm.reasoning_effort,
        context_manager=compiler._config.llm.context_manager,
        context_window=compiler._context_window.max_input,
        usage_tracker=compiler._usage_tracker,
        llm=compiler._config.llm,
    )

    # Per-batch retry: the agent already retries internally (via PydanticAI's
    # ``retries``), but this loop retries at the batch level when the entire
    # agent call returns None (e.g. transient timeout, empty response that
    # exhausts internal retries). ``compile.llm.batch_retry_attempts`` is
    # configurable — set to 0 to disable.
    batch_retries = compiler._config.llm.batch_retry_attempts
    validated: ExtractionOutput | None = None
    for attempt in range(1 + batch_retries):
        validated = try_pydanticai_extraction(
            agent,
            text=combined,
            source_id=source,
        )
        if validated is not None:
            break
        if attempt < batch_retries:
            logger.info(
                "TwoPhase: extraction batch retry %d/%d for '%s'",
                attempt + 1,
                batch_retries,
                source,
            )

    compiler._batch_tracker.record(failed=validated is None)
    if validated is None:
        logger.warning(
            "TwoPhase: extraction skipped for source '%s' — agent returned no result after %d batch attempt(s)",
            source,
            1 + batch_retries,
        )
        return None
    result = Extraction(
        source=source,
        summary=validated.summary,
        entities=list(validated.entities),
        concepts=list(validated.concepts),
        tags=validated.tags,
    )
    if not compiler._config.no_cache:
        compiler._cache_store.set("extract", extract_deps, result.to_dict())
    return result


def extract_incremental(
    compiler: TwoPhaseRuntime,
    source: str,
    batches_list: list[str],
    extraction_system: str,
    *,
    progress: ProgressFn | None = None,
    texts: list[str] | None = None,
) -> tuple[list[Extraction | None], bool]:
    """Run extraction with message_history threading between batches.

    Per-batch cache is checked first — cached batches skip the LLM call
    entirely. Only uncached batches are sent to the model, and the
    message history threads between consecutive uncached batches.

    History compaction: the threaded ``message_history`` is compacted
    when its estimated token cost exceeds ``context_window.history_budget``.
    The oldest (request, response) pair is dropped first, preserving
    the most recent context the model actually needs. Without this
    guard, long documents with many batches fill the model's context
    window and the model returns empty (``finish_reason: stop``, no
    content) — see commit history for the regression this fixed.

    *texts*, when given, are the source's text blocks the batches were packed
    from. A provider overflow then re-plans only the texts not yet extracted
    against the lowered window: the results and the threaded history so far
    are kept, and no text is sent twice.

    *progress* is called after every finished batch (cached ones
    included) so the bar moves with the work.
    """
    compaction_threshold = compiler._context_window.history_budget
    chars_per_token = compiler._config.llm.default_chars_per_token
    results: list[Extraction | None] = []
    history: list[ModelMessage] = []
    agent: ExtractionAgent | None = None
    # The unit of work is a text, not a batch: only a text list lets a re-plan
    # split the batch that overflowed without dropping or repeating content.
    pending: list[str] = list(texts) if texts is not None else list(batches_list)
    while pending:
        # Without the unbatched texts there is no text grouping to recover:
        # keep the batches the caller already packed.
        groups = _text_groups(pending, _extract_char_budget(compiler)) if texts is not None else [[batch] for batch in pending]
        extracted = 0
        try:
            for group in groups:
                combined = "\n\n".join(group)
                batch_idx = len(results)
                # Per-batch cache (same key as extract_one).
                extract_deps = _extract_cache_deps(compiler, combined)
                if not compiler._config.no_cache:
                    cached = compiler._cache_store.get_dict("extract", extract_deps)
                    if cached is not None:
                        logger.info("TwoPhase: extraction cache hit for '%s' batch %d", source, batch_idx)
                        extraction = Extraction.from_dict(cached)
                        results.append(extraction)
                        # Inject a synthetic summary of the cached result so the
                        # next uncached batch sees coherent cross-batch context
                        # (otherwise the agent loses the document thread).
                        history.append(_synthetic_history_response(extraction))
                        _compact_history_inplace(history, compaction_threshold, chars_per_token=chars_per_token)
                        extracted += len(group)
                        if progress is not None:
                            progress(len(results), len(groups), "")
                        continue
                    logger.info(
                        "TwoPhase: extraction cache miss for '%s' batch %d (text_hash=%.16s)",
                        source,
                        batch_idx,
                        extract_deps[CacheDeps.SOURCE_TEXT],
                    )

                if agent is None:
                    agent = ExtractionAgent(
                        model_string=compiler._model or "",
                        timeout=compiler._timeout,
                        skills=compiler._config.skills,
                        delta_retry_enabled=compiler._config.llm.delta_retry_enabled,
                        reasoning_effort=compiler._config.llm.reasoning_effort,
                        context_manager=compiler._config.llm.context_manager,
                        context_window=compiler._context_window.max_input,
                        usage_tracker=compiler._usage_tracker,
                        llm=compiler._config.llm,
                    )
                deps = PipelineAgentDeps(text=combined, source_id=f"{source}:{batch_idx}")
                try:
                    validated, new_messages = agent.run_with_messages(
                        deps,
                        message_history=history if history else None,
                    )
                    history = new_messages
                    _compact_history_inplace(history, compaction_threshold, chars_per_token=chars_per_token)
                    compiler._batch_tracker.record(failed=False)
                    result = Extraction(
                        source=source,
                        summary=validated.summary,
                        entities=list(validated.entities),
                        concepts=list(validated.concepts),
                        tags=validated.tags,
                    )
                    if not compiler._config.no_cache:
                        compiler._cache_store.set("extract", extract_deps, result.to_dict())
                    results.append(result)
                except LLMContextOverflowError:
                    # The batch did not fit the provider's window. The re-plan
                    # below lowers the window and splits the remaining texts -
                    # the error carries the limit the provider named.
                    raise
                except LLMOutputError:
                    logger.warning(
                        "TwoPhase: incremental extraction failed for '%s' batch %d",
                        source,
                        batch_idx,
                    )
                    compiler._batch_tracker.record(failed=True)
                    results.append(None)
                extracted += len(group)
                if progress is not None:
                    progress(len(results), len(groups), "")
                if compiler._batch_tracker.should_stop():
                    return results, True
        except LLMContextOverflowError as exc:
            if texts is None:
                raise
            # Keep every completed batch and the history it built; re-plan
            # only what this batch and the later ones cover.
            pending = _replan_after_overflow(compiler, source, pending[extracted:], exc)
            continue
        pending = []
    return results, False


def _replan_after_overflow(
    compiler: TwoPhaseRuntime,
    source: str,
    texts: list[str],
    exc: LLMContextOverflowError,
) -> list[str]:
    """Lower the window to the provider's limit and fit *texts* to it.

    Returns the texts that still have to be extracted, each already small
    enough for the re-planned budget - a text larger than the new window is
    split into windows instead of being sent whole again. The caller packs
    them into batches and keeps whatever it already extracted.

    The re-plan works on the source's *texts*, not the batches they were
    packed into: a joined batch has no text boundaries left to split on, so
    re-packing it could only hard-split the middle of a text.

    Raises the original error when the provider named no limit or the window
    is already at that limit - kompend cannot plan against an unknown limit,
    and re-planning an unchanged budget would produce the same request.
    """
    limit = exc.stated_limit
    if limit is None:
        raise exc
    if not compiler._clamp_window(limit, reason=f"provider enforces {limit} tokens"):
        raise exc
    char_budget = _extract_char_budget(compiler)
    fitted = _fit_texts(texts, char_budget)
    logger.info(
        "TwoPhase: re-planned '%s' into %d text window(s) for the %d-token window",
        source,
        len(fitted),
        compiler._context_window.max_input,
    )
    return fitted


def _fit_texts(texts: list[str], char_budget: int) -> list[str]:
    """Split any text that does not fit *char_budget* into windows.

    Batching can split a source between its texts, but a single text larger
    than the budget can only be split inside itself. ``split_text_windows``
    keeps every character (breaking on line boundaries where it can), so a
    chunk that grew past the window is still extracted in full - and a text
    that fits is passed through untouched.
    """
    fitted: list[str] = []
    for text in texts:
        if len(text) > char_budget:
            fitted.extend(split_text_windows(text, char_budget))
        elif text:
            fitted.append(text)
    return fitted


def _text_groups(texts: list[str], char_budget: int) -> list[list[str]]:
    """Pack *texts* into groups that fit within *char_budget* (chars).

    The groups keep the texts they contain (not the joined strings) so a
    caller that has to re-plan can tell exactly which texts a group covers.
    A text larger than the budget still forms its own group; callers that
    must fit the window split it first (:func:`_fit_texts`).
    """
    groups: list[list[str]] = []
    parts: list[str] = []
    length = 0
    for text in texts:
        if length + len(text) > char_budget and parts:
            groups.append(parts)
            parts = []
            length = 0
        parts.append(text)
        length += len(text)
    if parts:
        groups.append(parts)
    return groups


def _extract_char_budget(compiler: TwoPhaseRuntime) -> int:
    """Character budget for one extraction batch.

    The per-request cost that is NOT source content: the system template, the
    reserved output, and the shared context safety margin.  Every term is
    measured from the resolved config, never hard-coded, and the arithmetic
    lives in :class:`~kompend.llm.request_budget.RequestBudget` so every
    stage sizes itself the same way.

    The safety margin matters here: the chars-per-token heuristic undercounts
    token-dense text (formulas, tables), so a nominally exact-fit batch used to
    overflow a small local model's window — observed as
    ``request (65762 tokens) exceeds the available context size (65536)``.
    Content is never cropped: a smaller budget splits the source across more
    batches.
    """
    budget = RequestBudget(
        window=compiler._context_window,
        reserved={"system": compiler._extract_system_tokens},
    )
    logger.info("TwoPhase: extract budget (%s)", budget.report())
    return budget.content_chars(compiler._config.llm.default_chars_per_token)


def _extract_cache_deps(compiler: TwoPhaseRuntime, text: str) -> dict[str, str]:
    """Cache dependencies for one extraction batch.

    Single source of truth for the ``extract`` cache key - used by both
    :func:`extract_one` and :func:`extract_incremental`, which must agree.

    ``CacheDeps.MODEL`` is part of the key because extraction output is
    model-specific (which concepts a model finds, how it names them, how
    verbose the definitions are).  Without it, switching models replays
    the previous model's extractions instead of re-running them.  With
    ``cache_ignore_model`` the model is left out on purpose: the caller
    accepts replaying prior-model output to avoid re-running everything.
    """
    budget = compiler._budget
    deps = {
        CacheDeps.SOURCE_TEXT: hashlib.sha256(text.encode()).hexdigest(),
        CacheDeps.CHUNK_SIZE_MIN: str(budget.chunk_size_min),
        CacheDeps.CHUNK_SIZE_MAX: str(budget.chunk_size_max),
    }
    if not compiler._config.cache_ignore_model:
        deps[CacheDeps.MODEL] = compiler._model or ""
    return deps


def _compact_history_inplace(history: list[ModelMessage], max_tokens: int, *, chars_per_token: float) -> None:
    """Drop the oldest messages from ``history`` until under budget.

    Mutates the list in place. Messages come in (request, response) pairs
    in pydantic-ai, so dropping the two oldest preserves pairing. Keeps
    at least the last (request, response) pair so the agent always has
    some context to continue from.
    """
    if not history:
        return
    while _estimate_history_tokens(history, chars_per_token=chars_per_token) > max_tokens and len(history) > 2:
        # Drop the oldest request or response. Remove two at a time when
        # possible to keep request/response pairing intact.
        drop_n = 2 if len(history) > 2 else 1
        del history[:drop_n]
    if _estimate_history_tokens(history, chars_per_token=chars_per_token) > max_tokens and len(history) >= 2:
        logger.debug(
            "TwoPhase: history still over budget after compaction (%d tokens, %d messages) — keeping tail",
            _estimate_history_tokens(history, chars_per_token=chars_per_token),
            len(history),
        )


def _estimate_history_tokens(history: list[ModelMessage], *, chars_per_token: float) -> int:
    """Estimate total tokens across all messages in ``history``.

    Walks every ``ModelRequest`` / ``ModelResponse`` and sums the string
    length of each part's ``content`` field. Non-string parts (images,
    tool calls) are ignored — they're rare in the extraction path.

    The conversion uses the configured chars-per-token
    (``compile.llm.default_chars_per_token``), the same heuristic the batch
    budget is built from: measuring the history with a different ratio than
    the one the window was sized with is how a compacted history still
    overflows.
    """
    total = 0
    for msg in history:
        parts = getattr(msg, "parts", [])
        for part in parts:
            content = getattr(part, "content", None)
            if isinstance(content, str):
                total += chars_to_tokens(len(content), chars_per_token=chars_per_token)
    return total


def _synthetic_history_response(extraction: Extraction) -> ModelResponse:
    """Build a synthetic ``ModelResponse`` summarising a cached extraction.

    When a batch is served from the per-batch cache, we still want the
    next uncached batch to see what concepts/entities were extracted
    upstream. Rather than replaying the original (potentially huge)
    assistant turn, we inject a compact one-line summary.
    """
    concepts = ", ".join(c.name or "?" for c in extraction.concepts[:LOG_SAMPLE_LIMIT])
    entities = ", ".join(e.name or "?" for e in extraction.entities[:LOG_SAMPLE_LIMIT])
    summary_bits = [f"summary: {extraction.summary}"]
    if concepts:
        summary_bits.append(f"concepts: {concepts}")
    if entities:
        summary_bits.append(f"entities: {entities}")
    return ModelResponse(parts=[TextPart(content=" | ".join(summary_bits))])


def combine_extractions(compiler: TwoPhaseRuntime, source: str, parts: list[Extraction]) -> Extraction:
    """Combine per-batch extractions into one document-level Extraction."""
    if not parts:
        return Extraction(source=source)
    if len(parts) == 1:
        return parts[0]
    strategy = compiler._config.llm.extraction_strategy
    if strategy == ExtractionStrategy.REDUCE:
        return compiler._reduce_extractions(source, parts)
    return merge_extractions(source, parts)


def reduce_extractions(compiler: TwoPhaseRuntime, source: str, parts: list[Extraction]) -> Extraction:
    """LLM-powered reduce: merge partial extractions into a coherent whole."""
    partial_data = encode_payload([p.to_dict() for p in parts], compiler._config.llm.data_format)
    reduce_deps_key: dict[str, str] = {
        CacheDeps.SOURCE_TEXT: hashlib.sha256(partial_data.encode()).hexdigest(),
        CacheDeps.STAGE: "reduce",
    }
    if not compiler._config.cache_ignore_model:
        reduce_deps_key[CacheDeps.MODEL] = compiler._model
    if not compiler._config.no_cache:
        cached = compiler._cache_store.get_dict("extract", reduce_deps_key)
        if cached is not None:
            return Extraction.from_dict(cached)

    agent = ExtractionReduceAgent(
        model_string=compiler._model or "",
        timeout=compiler._timeout,
        skills=compiler._config.skills,
        reasoning_effort=compiler._config.llm.reasoning_effort,
        context_window=compiler._context_window.max_input,
        usage_tracker=compiler._usage_tracker,
        llm=compiler._config.llm,
    )
    try:
        deps = PipelineAgentDeps(text=partial_data, source_id=f"{source}:reduce")
        validated = agent.run(deps)
    except LLMOutputError:
        logger.warning("TwoPhase: reduce failed for '%s' — falling back to naive merge", source)
        return merge_extractions(source, parts)
    result = Extraction(
        source=source,
        summary=validated.summary,
        entities=list(validated.entities),
        concepts=list(validated.concepts),
        tags=validated.tags,
    )
    if not compiler._config.no_cache:
        compiler._cache_store.set("extract", reduce_deps_key, result.to_dict())
    return result
