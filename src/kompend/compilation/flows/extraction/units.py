"""Extraction units — batch planning, budget, overflow re-planning, and calls.

Unit selection and thresholds, the per-unit LLM call with its retry policy,
and the merge/reduce of per-unit extractions. Pure planning lives here so the
orchestration in :mod:`flow` stays readable.
"""

from __future__ import annotations

import hashlib

from pydantic_ai.messages import ModelMessage

from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.extraction import ExtractionAgent
from kompend.agents.extraction_reduce import ExtractionReduceAgent
from kompend.agents.failure import UnitFailure, UnitFailureLog, is_retry_pointless, record_unit_failure
from kompend.compilation.flows.extraction.prompt import (
    _compact_history_inplace,
    _synthetic_history_response,
)
from kompend.compilation.schema import ExtractionOutput
from kompend.compilation.two_phase import (
    Extraction,
    ProgressFn,
    TwoPhaseRuntime,
    merge_extractions,
    try_pydanticai_extraction,
)
from kompend.core.config._configs import ExtractionStrategy
from kompend.core.models.enums import CacheDeps
from kompend.core.text.windows import split_text_windows
from kompend.core.utils.exceptions import LLMContextOverflowError, LLMOutputError
from kompend.llm.format import encode_payload
from kompend.llm.request_budget import RequestBudget
from kompend.logging import get_logger

logger = get_logger(__name__)


def batch_texts(texts: list[str], char_budget: int) -> list[str]:
    """Group texts into batches that fit within *char_budget* (chars)."""
    return ["\n\n".join(group) for group in _text_groups(texts, char_budget)]


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
    # ``retries``), but this loop retries at the batch level when the agent
    # returned no result at all. ``compile.llm.batch_retry_attempts`` is
    # configurable — set to 0 to disable. The retry is skipped when repeating
    # the identical request cannot change the answer: an empty answer or a
    # request too large for the window (``is_retry_pointless``).
    batch_retries = compiler._config.llm.batch_retry_attempts
    failures = UnitFailureLog(stage="extraction")
    validated: ExtractionOutput | None = None
    last_failure: UnitFailure | None = None
    for attempt in range(1 + batch_retries):
        before = len(failures.failures)
        validated = try_pydanticai_extraction(
            agent,
            text=combined,
            source_id=source,
            failures=failures,
        )
        if validated is not None:
            break
        if len(failures.failures) > before:
            last_failure = failures.failures[-1]
        if last_failure is not None and is_retry_pointless(last_failure.kind):
            logger.info(
                "TwoPhase: not retrying extraction for '%s' — %s cannot be fixed by repeating the request",
                source,
                last_failure.kind.value,
            )
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
        cause = last_failure.describe() if last_failure is not None else "no result"
        logger.warning(
            "TwoPhase: extraction skipped for source '%s' — %s",
            source,
            cause,
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
                except Exception as exc:  # classified into the failure vocabulary below
                    record_unit_failure(stage="extraction:incremental", unit=f"{source}:{batch_idx}", exc=exc)
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
    token-dense text (formulas, tables), so a nominally exact-fit batch would
    overflow a small local model's window
    (``request (65762 tokens) exceeds the available context size (65536)``).
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
    except LLMOutputError as exc:
        # Reducing is best-effort enrichment: the per-batch extractions are
        # already complete, so the merge below loses nothing but dedup.
        failure = record_unit_failure(stage="extraction:reduce", unit=source, exc=exc)
        logger.warning("TwoPhase: reduce failed for '%s' (%s) — falling back to naive merge", source, failure.kind.value)
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
