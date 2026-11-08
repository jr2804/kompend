"""Stall-watchdog behaviour: one warning per silent item, stage attributed.

The watchdog (knox-ty8f.8) enforces nothing — it names the file, the stage,
and the silence so a wedged backend is distinguishable from a slow one while
the progress bar sits still. Enforcement stays with the parse deadline
(test_ingest_deadline); here we prove the watchdog's own contract: exactly
one warning per item, suppressed by heartbeats, and the deadline row carries
the stage the file died in.
"""

from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path
from typing import cast

import pytest

from kompend.core.models import ParsedDocument
from kompend.core.pipeline.bounded_parse import ParseDeadlineExceeded
from kompend.core.pipeline.ingest import ingest_documents
from kompend.core.pipeline.stall_watch import DEFAULT_STALL_WARN_S, StallWatchdog, WatchStage
from kompend.core.types import ReasonCode, StageOutcome
from kompend.preprocessing.parsers import DocumentParser

_LOGGER_NAME = "kompend.core.pipeline.stall_watch"

_ingest = importlib.import_module("kompend.core.pipeline.ingest")


class _FakeClock:
    """Manual clock: tests advance it instead of sleeping."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _SlowParser:
    """Picklable stand-in that takes its time on one file."""

    def __init__(self, delay_s: float) -> None:
        self._delay_s = delay_s

    def parse(self, target: Path) -> ParsedDocument:
        time.sleep(self._delay_s)
        return ParsedDocument(source_file=Path(target), elements=[], title=Path(target).stem)


class _SlowChain:
    """Parser façade routed per file; mirrors the deadline-test harness."""

    def __init__(self, delay_s: float) -> None:
        self._delay_s = delay_s
        self.parsed: list[str] = []

    def backend_identity(self) -> str:
        return "stall-watch-chain"

    def route(self, source_file: Path) -> _SlowParser:
        slow = Path(source_file).name == "stuck.md"
        return _SlowParser(self._delay_s if slow else 0.0)


class _DeadlineParser:
    """Façade whose stuck file raises the deadline error, as in the deadline test."""

    def __init__(self) -> None:
        self.parsed: list[str] = []

    def backend_identity(self) -> str:
        return "stall-attribution-chain"

    def route(self, source_file: Path) -> _DeadlineBackend:
        return _DeadlineBackend(self.parsed)


class _DeadlineBackend:
    def __init__(self, seen: list[str]) -> None:
        self.seen = seen

    def parse(self, target: Path) -> ParsedDocument:
        self.seen.append(Path(target).name)
        return ParsedDocument(source_file=Path(target), elements=[], title=Path(target).stem)


def test_default_threshold_is_thirty_seconds() -> None:
    assert DEFAULT_STALL_WARN_S == 30.0


def test_scan_warns_once_naming_stage_source_and_elapsed(caplog: pytest.LogCaptureFixture) -> None:
    clock = _FakeClock()
    watchdog = StallWatchdog(10.0, clock=clock)
    watchdog.watch("notes/stuck.md", stage=WatchStage.PARSE)
    clock.advance(12.5)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        first = watchdog._scan()
        second = watchdog._scan()

    assert first == [("notes/stuck.md", WatchStage.PARSE, 12.5)]
    assert second == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "no progress for 12.5 s in stage parse: notes/stuck.md" in warnings[0].getMessage()
    watchdog.close()


def test_a_beat_resets_the_silence_window_and_advances_the_stage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _FakeClock()
    watchdog = StallWatchdog(10.0, clock=clock)
    watchdog.watch("a.md", stage=WatchStage.PARSE)
    clock.advance(8.0)
    watchdog.beat("a.md", stage=WatchStage.POSTPROCESS)
    clock.advance(8.0)  # 16 s since watch, but only 8 s since the last beat

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        assert watchdog._scan() == []
        assert caplog.records == []

    assert watchdog.last_stage("a.md") is WatchStage.POSTPROCESS
    watchdog.close()


def test_unwatched_items_never_warn(caplog: pytest.LogCaptureFixture) -> None:
    clock = _FakeClock()
    watchdog = StallWatchdog(10.0, clock=clock)
    watchdog.watch("done.md", stage=WatchStage.PARSE)
    watchdog.unwatch("done.md")
    clock.advance(1000.0)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        assert watchdog._scan() == []
        assert caplog.records == []

    assert watchdog.last_stage("done.md") is None
    watchdog.close()


def test_zero_budget_disables_the_watchdog() -> None:
    watchdog = StallWatchdog(0.0)
    watchdog.start()  # must not start an observer thread
    watchdog.watch("a.md", stage=WatchStage.PARSE)
    assert watchdog.last_stage("a.md") is None
    watchdog.beat("a.md", stage=WatchStage.CHUNK)
    watchdog.unwatch("a.md")
    watchdog.close()


def test_last_stage_tracks_the_latest_stage_and_ignores_strangers() -> None:
    clock = _FakeClock()
    watchdog = StallWatchdog(10.0, clock=clock)
    watchdog.watch("a.md", stage=WatchStage.PARSE)
    watchdog.beat("a.md", stage=WatchStage.CHUNK)
    watchdog.beat("a.md", stage=WatchStage.ENRICH)

    assert watchdog.last_stage("a.md") is WatchStage.ENRICH
    assert watchdog.last_stage("never-watched.md") is None
    watchdog.close()


def test_ingest_warns_once_for_a_stuck_source_and_lets_the_run_finish(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = _seed_vault(tmp_path)
    stuck = base / "sources" / "stuck.md"
    fine = base / "sources" / "fine.md"
    stuck.write_text("# stuck\n", encoding="utf-8")
    fine.write_text("# fine\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = ingest_documents(
            files=[stuck, fine],
            base_dir=base,
            parser=cast("DocumentParser", _SlowChain(delay_s=0.6)),
            reparse=True,
            config=None,
            parse_timeout_s=0.0,  # watchdog only; no deadline kills
            stall_warn_s=0.15,
        )

    assert result.count == 2
    assert result.failed == 0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "stuck.md" in message
    assert "stage parse" in message


def test_deadline_row_names_the_stage_the_file_died_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _seed_vault(tmp_path)
    stuck = base / "sources" / "stuck.md"
    fine = base / "sources" / "fine.md"
    stuck.write_text("# stuck\n", encoding="utf-8")
    fine.write_text("# fine\n", encoding="utf-8")

    def _parse(
        self: object,
        parser: _DeadlineBackend,
        target: Path,
        *,
        source: str | None = None,  # noqa: ARG001 — signature parity
    ) -> ParsedDocument:
        if Path(target).name == "stuck.md":
            raise ParseDeadlineExceeded(Path(target), 5.0)
        return parser.parse(target)

    monkeypatch.setattr(_ingest.DeadlineWorker, "parse", _parse)

    rows: list[tuple[str, StageOutcome, str]] = []
    real_record = _ingest.StageLedger.record

    def _record(
        self: _ingest.StageLedger,
        source: str,
        stage: str,
        *,
        outcome: StageOutcome,
        attempt: int = 1,
        duration_ms: float | None = None,
        reason_code: ReasonCode | None = None,
        detail: str | None = None,
        parser: str | None = None,
        parser_version: str | None = None,
        backend: str | None = None,
        judge_model: str | None = None,
        cache_key: str | None = None,
        sha: str | None = None,
        size_bytes: int | None = None,
        run_id: str | None = None,
    ) -> object:
        rows.append((str(source), outcome, detail or ""))
        return real_record(
            self,
            source,
            stage,
            outcome=outcome,
            attempt=attempt,
            duration_ms=duration_ms,
            reason_code=reason_code,
            detail=detail,
            parser=parser,
            parser_version=parser_version,
            backend=backend,
            judge_model=judge_model,
            cache_key=cache_key,
            sha=sha,
            size_bytes=size_bytes,
            run_id=run_id,
        )

    monkeypatch.setattr(_ingest.StageLedger, "record", _record)

    result = ingest_documents(
        files=[stuck, fine],
        base_dir=base,
        parser=cast("DocumentParser", _DeadlineParser()),
        reparse=True,
        config=None,
        parse_timeout_s=5.0,
        stall_warn_s=1.0,
    )

    assert result.failed == 1
    timeout_rows = [row for row in rows if row[1] is StageOutcome.TIMEOUT]
    assert len(timeout_rows) == 1
    source, _, detail = timeout_rows[0]
    assert source == "sources/stuck.md"
    assert detail.endswith("Last progress: stage parse.")


def _seed_vault(tmp_path: Path) -> Path:
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".kompend").mkdir(parents=True)
    (base / ".kompend" / "config.toml").write_text("", encoding="utf-8")
    return base
