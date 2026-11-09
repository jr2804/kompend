"""The parse memory budget: plan-aware headroom, serialization, config surface."""

from __future__ import annotations

import threading
from dataclasses import fields
from pathlib import Path

import pytest

from kompend.core.config._configs import KompendConfig
from kompend.core.env import EnvVar
from kompend.core.pipeline.config import PipelineConfig
from kompend.preprocessing import host_memory
from kompend.preprocessing.host_memory import (
    HEAVY_SUFFIXES,
    LIGHT_BACKENDS,
    MEGABYTE,
    HeavyParseGuard,
    InsufficientHostMemoryError,
    ensure_engine_headroom,
    heavy_backends,
    is_heavy_source,
)
from kompend.preprocessing.parsers.base import FileExtension
from kompend.preprocessing.parsers.options import ParserOptions

# ── serialization classification ─────────────────────────────────────────
#
# The set is derived from FileExtension, so the cases that matter are the
# boundaries: a binary container is serialized, a text format never is —
# including the ones a hand-written list got wrong (.htm/.html are native text).


@pytest.mark.parametrize(
    ("name", "heavy"),
    [
        ("report.pdf", True),
        ("deck.pptx", True),
        ("scan.png", True),
        ("note.md", False),
        ("page.htm", False),
        ("page.html", False),
        ("source.py", False),
        ("unknown.z9", False),  # unparseable today: no parser reaches the throttle
    ],
)
def test_classification_matches_the_extension_table(name: str, heavy: bool) -> None:
    assert is_heavy_source(Path(name)) is heavy


def test_heavy_suffixes_are_derived_not_declared() -> None:
    assert frozenset(e.value for e in FileExtension if not e.is_text) == HEAVY_SUFFIXES


# ── plan-aware headroom ──────────────────────────────────────────────────
#
# Which engine a source reaches is the router's decision, so the budget must
# follow the plan: a PDF liteparse reads needs a few hundred megabytes, the
# same PDF routed to docling needs gigabytes. Classifying by suffix refused
# both and skipped every PDF in a liteparse-only vault.


@pytest.mark.parametrize("planned", ["liteparse", "native-markdown", "pymupdf"])
def test_a_light_plan_never_checks_the_host(monkeypatch: pytest.MonkeyPatch, planned: str) -> None:
    def _unexpected() -> int | None:
        raise AssertionError("a light plan must not ask the host")

    monkeypatch.setattr("kompend.preprocessing.host_memory.available_commit_bytes", _unexpected)
    ensure_engine_headroom(Path("report.pdf"), [planned], min_free_commit_bytes=4096 * MEGABYTE)


@pytest.mark.parametrize("planned", ["docling", "mineru", "marker", "ocr-surya", "a-backend-added-later"])
def test_an_unmeasured_or_heavy_plan_is_guarded(monkeypatch: pytest.MonkeyPatch, planned: str) -> None:
    """``LIGHT_BACKENDS`` is a positive list: everything else is heavy until measured."""
    monkeypatch.setattr("kompend.preprocessing.host_memory.available_commit_bytes", lambda: 512 * MEGABYTE)
    with pytest.raises(InsufficientHostMemoryError):
        ensure_engine_headroom(Path("report.pdf"), [planned], min_free_commit_bytes=4096 * MEGABYTE)


def test_heavy_backends_names_only_the_heavy_part_of_a_plan() -> None:
    assert heavy_backends(["liteparse", "docling", "native-text"]) == frozenset({"docling"})
    assert "liteparse" in LIGHT_BACKENDS


def test_the_refusal_names_the_plan_the_shortfall_and_the_levers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kompend.preprocessing.host_memory.available_commit_bytes", lambda: 512 * MEGABYTE)
    with pytest.raises(InsufficientHostMemoryError) as exc_info:
        ensure_engine_headroom(Path("report.pdf"), ["docling"], min_free_commit_bytes=4096 * MEGABYTE)
    message = str(exc_info.value)
    assert "512 MB" in message
    assert "4096 MB" in message
    assert "docling" in message
    assert "1830 MB" in message  # the measured peak, not a guess
    assert "[ingest] parse_min_free_commit_mb" in message
    assert "[ingest] parsecraft_docling_threads = 1" in message


def test_a_non_docling_engine_gets_only_the_levers_that_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kompend.preprocessing.host_memory.available_commit_bytes", lambda: 1)
    with pytest.raises(InsufficientHostMemoryError) as exc_info:
        ensure_engine_headroom(Path("scan.pdf"), ["mineru"], min_free_commit_bytes=MEGABYTE)
    message = str(exc_info.value)
    assert "docling" not in message
    assert "[ingest] parse_min_free_commit_mb" in message


def test_zero_disables_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected() -> int | None:
        raise AssertionError("a disabled check must not ask the host")

    monkeypatch.setattr("kompend.preprocessing.host_memory.available_commit_bytes", _unexpected)
    ensure_engine_headroom(Path("report.pdf"), ["docling"], min_free_commit_bytes=0)


def test_an_unknown_platform_warns_once_and_proceeds(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    host_memory._warn_unknown_headroom_once.cache_clear()
    monkeypatch.setattr("kompend.preprocessing.host_memory.available_commit_bytes", lambda: None)
    with caplog.at_level("WARNING"):
        for _ in range(2):
            ensure_engine_headroom(Path("report.pdf"), ["docling"], min_free_commit_bytes=4096 * MEGABYTE)
    warnings = [r for r in caplog.records if "unknown" in r.getMessage()]
    assert len(warnings) == 1


# ── serialization ────────────────────────────────────────────────────────


def test_heavy_parses_are_serialized_at_max_concurrent_one() -> None:
    guard = HeavyParseGuard(max_concurrent=1)
    inside = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def _first() -> None:
        with guard.guard(Path("a.pdf")):
            inside.set()
            release.wait(timeout=5)
            order.append("first")

    thread = threading.Thread(target=_first)
    thread.start()
    assert inside.wait(timeout=5)
    # The second heavy parse only enters after the first one leaves the guard.
    started = threading.Event()

    def _second() -> None:
        with guard.guard(Path("b.pdf")):
            started.set()
            order.append("second")

    thread2 = threading.Thread(target=_second)
    thread2.start()
    thread2.join(timeout=0.2)
    assert not started.is_set()
    release.set()
    thread.join(timeout=5)
    thread2.join(timeout=5)
    assert order == ["first", "second"]


def test_a_text_source_never_takes_a_slot() -> None:
    guard = HeavyParseGuard(max_concurrent=1)
    with guard.guard(Path("a.pdf")), guard.guard(Path("note.md")):  # would deadlock if it took the semaphore
        pass


# ── config surface ───────────────────────────────────────────────────────


def test_the_config_fields_exist_with_the_measured_defaults() -> None:
    config = KompendConfig()
    # Four threads is the measured sweet spot: the peak drops from 2790 MB at
    # the engine's own count to 1830 MB, for 0.3 s more per 8-page document.
    assert config.ingest.parsecraft_docling_threads == 4
    assert config.ingest.parse_min_free_commit_mb == 4096
    assert config.ingest.max_concurrent_heavy_parses == 1


def test_the_parser_defaults_to_no_budget_and_the_pipeline_passes_it_per_run() -> None:
    # A parser built without config (extension discovery) must not refuse
    # anything; the pipeline's parser carries the budget from ParserOptions.
    assert ParserOptions().parse_min_free_commit_mb == 0
    assert next(f for f in fields(PipelineConfig) if f.name == "max_concurrent_heavy_parses").default == 1


def test_the_env_vars_are_declared() -> None:
    assert EnvVar.INGEST_PARSECRAFT_DOCLING_THREADS == "KOMP_INGEST_PARSECRAFT_DOCLING_THREADS"
    assert EnvVar.INGEST_PARSE_MIN_FREE_COMMIT_MB == "KOMP_INGEST_PARSE_MIN_FREE_COMMIT_MB"
    assert EnvVar.INGEST_MAX_CONCURRENT_HEAVY_PARSES == "KOMP_INGEST_MAX_CONCURRENT_HEAVY_PARSES"
