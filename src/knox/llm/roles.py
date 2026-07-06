"""Centralised LLM call-role identifiers.

Every LLM call in KNOX is tagged with a *role* (used for per-role failover
chains via ``compile.llm.role_models`` and for log/metric attribution).
Historically these were hard-coded string literals scattered across modules
— a forbidden antipattern (schema field-name literals). This StrEnum is the
single source of truth: import from here, never write the string inline.
"""

from __future__ import annotations

from enum import StrEnum


class LLMRole(StrEnum):
    """All LLM call roles used across the pipeline."""

    DEFAULT = "default"

    # ── Capture ──────────────────────────────────────────────────────────
    CAPTURE = "capture"
    CAPTURE_DISTILL = "capture.distill"

    # ── Chunking ─────────────────────────────────────────────────────────
    CHUNKING_CONTEXTUAL = "chunking.contextual"

    # ── Compilation (two-phase + WiCER) ──────────────────────────────────
    COMPILE_EXTRACT = "compile.extract"
    COMPILE_GENERATE = "compile.generate"
    WICER_PROBE = "wicer.probe"
    WICER_JUDGE = "wicer.judge"
    WICER_REFINE = "wicer.refine"

    # ── Lint (LLM-backed rules + remediation) ────────────────────────────
    LINT_CLARITY_SCORE = "lint.clarity-score"
    LINT_CONTRADICTION = "lint.contradiction-detection"
    LINT_COVERAGE_GAP = "lint.coverage-gap"
    LINT_COVERAGE_GAP_DEDUP = "lint.coverage-gap-dedup"
    LINT_FACTUAL_CONSISTENCY = "lint.factual-consistency"
    LINT_REMEDIATE_CONCEPT = "lint.remediate-concept"

    # ── Organize ─────────────────────────────────────────────────────────
    ORGANIZE_CATEGORIZE = "organize.categorize"

    # ── Search / RAG ─────────────────────────────────────────────────────
    SEARCH_CRAG = "search.crag"
    SEARCH_EXPAND = "search.expand"
    SEARCH_HYDE = "search.hyde"
    SYNTHESIZE = "synthesize"

    # ── Trust verification ───────────────────────────────────────────────
    VERIFY = "verify"

    # ── Doctor ───────────────────────────────────────────────────────────
    DOCTOR = "doctor"

    # ── Resolution (VLM image enrichment) ─────────────────────────────────
    VLM = "vlm"
