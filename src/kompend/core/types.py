"""Cross-cutting vocabularies shared by the pipeline and its observers.

This module is the home for small, stable enums that several subsystems
need to agree on — today the failure-reason vocabulary of the item path
(:class:`ReasonCode`), tomorrow the observability ledger that counts it.
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
    #: Stored state (parse cache, asset/reference rows, attachment bytes)
    #: failed validation and was treated as a miss.
    CORRUPT_CACHE = "corrupt_cache"
    #: No parser backend claims the item (unrouted format, VLM not
    #: configured, unsupported attachment type).
    NO_PARSER = "no_parser"
    #: The parser/VLM ran but produced nothing usable.
    EMPTY_EXTRACTION = "empty_extraction"
