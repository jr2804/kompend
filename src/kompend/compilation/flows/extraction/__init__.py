"""Extraction flow package — LLM extraction from source documents.

- :mod:`flow` — the orchestration ``stages/`` calls: source grouping,
  strategy dispatch, progress reporting, extraction stats
- :mod:`units` — unit selection and thresholds: text grouping, window
  budgeting, overflow re-planning, the per-unit call with its retry
  policy, and merge/reduce of per-unit extractions
- :mod:`prompt` — threaded-context assembly for the incremental path:
  history token estimation, in-place compaction, synthetic responses
"""

from kompend.compilation.flows.extraction.flow import (
    _compact_history_inplace,
    _estimate_history_tokens,
    _extract_char_budget,
    _source_progress,
    _synthetic_history_response,
    batch_texts,
    combine_extractions,
    extract_incremental,
    extract_one,
    extraction_detail,
    reduce_extractions,
    run_extraction,
)

__all__ = [
    "_compact_history_inplace",
    "_estimate_history_tokens",
    "_extract_char_budget",
    "_source_progress",
    "_synthetic_history_response",
    "batch_texts",
    "combine_extractions",
    "extract_incremental",
    "extract_one",
    "extraction_detail",
    "reduce_extractions",
    "run_extraction",
]
