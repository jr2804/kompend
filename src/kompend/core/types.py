"""Cross-cutting vocabularies shared by the pipeline and its observers.

This module is the home for small, stable enums that several subsystems
need to agree on — today the failure-reason vocabulary of the item path
(:class:`ReasonCode`) and the ledger's stage-outcome vocabulary
(:class:`StageOutcome`).
Anything imported from here must stay stdlib-only and import-cheap: the
store ledger, CLI diagnostics, and the parser stack all import it.
Domain-model vocabularies (element types, metadata fields) live in
``kompend.core.models.enums`` instead — that module is about data shapes,
this one is about runtime communication.
"""

from __future__ import annotations

from enum import StrEnum


class ReasonCode(StrEnum):
    """Stable vocabulary for item-path failures and degradations.

    Every item-path ``logger.error``/``logger.warning`` and every ledger row
    carries one of these values as ``reason_code`` (logger ``extra=...``), so
    downstream observability (ledger, ``kompend explain``, doctor) can count
    and filter incidents by class instead of pattern-matching prose.

    Members are public API: the per-item ledger, repro bundles and doctor
    screens reference them by name. Add members freely; never rename or
    recycle one — persisted rows carry the value verbatim.
    """

    #: A file's extension claims one format but the bytes are another
    #: (e.g. ``.md`` holding binary).
    MISLABELLED_BINARY = "mislabelled_binary"
    #: A parse exceeded its deadline (wedged backend) or timed out.
    TIMEOUT = "timeout"
    #: The parse backend could not run: unavailable, out of host memory, or
    #: failed in an unclassified way (the exception text stays in the log).
    BACKEND_UNAVAILABLE = "backend_unavailable"
    #: The parse-quality judge is unavailable and a deterministic fallback
    #: degraded the decision (decision tier).
    JUDGE_DEGRADED = "judge_degraded"
    #: A decision tier had to fall back to the lower tier (e.g. clef down).
    DECISION_FALLBACK = "decision_fallback"
    #: A summarizer traversal plan was made for a chunk tree (``summarize``
    #: stage, one row per tree). The full plan JSON goes to the
    #: ``summarizer_plan`` cache layer, not into the ledger detail.
    TRAVERSAL = "traversal"
    #: Stored state (parse cache, asset/reference rows, attachment bytes)
    #: failed validation and was treated as a miss.
    CORRUPT_CACHE = "corrupt_cache"
    #: No parser backend claims the item (unrouted format, VLM not
    #: configured, unsupported attachment type).
    NO_PARSER = "no_parser"
    #: The parser/VLM ran but produced nothing usable.
    EMPTY_EXTRACTION = "empty_extraction"
    #: The figure pre-filter dropped the item before the VLM ran (byte or
    #: dimension floor, or the structural/pixel heuristic); the ``detail``
    #: carries the specific reason.
    VLM_FILTERED = "vlm_filtered"
    #: The text-only relevance gate judged the figure decorative; the
    #: caption was deliberately skipped.
    VLM_RELEVANCE_SKIPPED = "vlm_relevance_skipped"
    #: The vision model itself answered SKIP for the figure.
    VLM_SKIPPED = "vlm_skipped"
    #: The concept-embedding merge fell back to string similarity; the
    #: warning is log-only (no ledger row).
    EMBEDDING_MERGE_DISABLED = "embedding_merge_disabled"
    #: A session payload exceeded the provider's effective token limit
    #: (input window or output ceiling); the caller split the batch and
    #: retried instead of re-asking byte-identical work.
    PAYLOAD_TOO_LARGE = "payload_too_large"


class StageOutcome(StrEnum):
    """What happened to one item in one pipeline stage (the ``stages`` ledger).

    One row per (run, source, stage, attempt) in the per-item ledger
    (:mod:`kompend.core.reporting.ledger`) carries exactly one of these as
    ``outcome``. ``failed`` vs ``timeout`` split the parse deadline from other
    failures; ``degraded`` means the item completed through a worse path
    (empty VLM caption kept as-is, judge fallback) rather than failing.

    Members are public API — persisted rows carry the value verbatim; add
    freely, never rename or recycle.
    """

    #: The stage completed normally.
    OK = "ok"
    #: A current cache entry satisfied the stage without recompute.
    CACHED = "cached"
    #: The item was deliberately not processed (no backend, dedup, disabled).
    SKIPPED = "skipped"
    #: The stage raised; the run continued with the remaining items.
    FAILED = "failed"
    #: The stage exceeded its wall-clock budget (``ingest.parse_timeout_s``).
    TIMEOUT = "timeout"
    #: The stage completed via a fallback path with reduced quality.
    DEGRADED = "degraded"
    #: A planned subset completed; the rest fell back to the default path
    #: (e.g. a traversal plan covered some levels while the others
    #: summarized whole).
    PARTIAL = "partial"
