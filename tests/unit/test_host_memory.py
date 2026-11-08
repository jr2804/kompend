"""The heavy-parse guard: classification, headroom refusal, serialization."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from kompend.core.config._configs import KompendConfig
from kompend.core.config.models.ingest import _DEFAULT_DOCLING_THREADS
from kompend.core.env import EnvVar
from kompend.core.pipeline.config import PipelineConfig
from kompend.preprocessing.host_memory import (
    HEAVY_SUFFIXES,
    MEGABYTE,
    HeavyParseGuard,
    InsufficientHostMemoryError,
    is_heavy_source,
)
from kompend.preprocessing.parsers.base import FileExtension

# ── classification ───────────────────────────────────────────────────────
#
# The set is derived from FileExtension, so the cases that matter are the
# boundaries: a binary container is heavy, a text format never is — including
# the ones a hand-written list got wrong (.htm/.html are native text).


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
        ("unknown.z9", False),  # unparseable today: no parser reaches the guard
    ],
)
def test_classification_matches_the_extension_table(name: str, heavy: bool) -> None:
    assert is_heavy_source(Path(name)) is heavy


def test_heavy_suffixes_are_derived_not_declared() -> None:

    assert frozenset(e.value for e in FileExtension if not e.is_text) == HEAVY_SUFFIXES


def test_a_heavy_parse_below_the_threshold_is_refused_with_the_remedy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kompend.preprocessing.host_memory.available_commit_bytes", lambda: 512 * MEGABYTE)
    guard = _guard(4096)
    with pytest.raises(InsufficientHostMemoryError) as exc_info, guard.guard(Path("report.pdf")):
        pass
    message = str(exc_info.value)
    assert "512 MB free" in message
    assert "4096 MB required" in message
    assert "parse_min_free_commit_mb" in message
    assert "parsecraft_docling_threads" in message


def test_zero_disables_the_headroom_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kompend.preprocessing.host_memory.available_commit_bytes", lambda: 1)
    with HeavyParseGuard().guard(Path("report.pdf")):  # min_free_commit_bytes=0
        pass  # not refused: the cap is the only protection left


def test_an_unknown_platform_warns_once_and_proceeds(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr("kompend.preprocessing.host_memory.available_commit_bytes", lambda: None)
    guard = _guard(4096)
    with caplog.at_level("WARNING"):
        for _ in range(2):
            with guard.guard(Path("report.pdf")):
                pass
    warnings = [r for r in caplog.records if "unknown" in r.getMessage()]
    assert len(warnings) == 1


def test_a_light_source_never_checks_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected() -> int | None:
        raise AssertionError("the light path must not ask the host")

    monkeypatch.setattr("kompend.preprocessing.host_memory.available_commit_bytes", _unexpected)
    with _guard(4096).guard(Path("note.md")):
        pass


# ── headroom ─────────────────────────────────────────────────────────────


def _guard(min_free_mb: int) -> HeavyParseGuard:
    return HeavyParseGuard(min_free_commit_bytes=min_free_mb * MEGABYTE)


# ── serialization ────────────────────────────────────────────────────────


def test_heavy_parses_are_serialized_at_max_concurrent_one() -> None:
    guard = HeavyParseGuard(min_free_commit_bytes=0, max_concurrent=1)
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


# ── config surface ───────────────────────────────────────────────────────


def test_the_config_fields_exist_with_the_measured_defaults() -> None:
    config = KompendConfig()
    assert config.ingest.parsecraft_docling_threads == _DEFAULT_DOCLING_THREADS
    assert 1 <= config.ingest.parsecraft_docling_threads <= 4
    assert config.ingest.parse_min_free_commit_mb == 10240
    assert config.ingest.max_concurrent_heavy_parses == 1


def test_the_pipeline_config_carries_the_guard_budget() -> None:
    # The dataclass defaults mirror [ingest]; ``from_resolved`` maps the
    # KompendConfig fields onto them (see the mapping block in config.py).
    assert PipelineConfig.parse_min_free_commit_mb == 10240
    assert PipelineConfig.max_concurrent_heavy_parses == 1


def test_the_env_vars_are_declared() -> None:
    assert EnvVar.INGEST_PARSECRAFT_DOCLING_THREADS == "KOMP_INGEST_PARSECRAFT_DOCLING_THREADS"
    assert EnvVar.INGEST_PARSE_MIN_FREE_COMMIT_MB == "KOMP_INGEST_PARSE_MIN_FREE_COMMIT_MB"
    assert EnvVar.INGEST_MAX_CONCURRENT_HEAVY_PARSES == "KOMP_INGEST_MAX_CONCURRENT_HEAVY_PARSES"
