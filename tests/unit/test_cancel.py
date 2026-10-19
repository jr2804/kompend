"""Tests for cooperative cancellation: CancelState, ESC watcher, and wiring.

Covers the cancel token, the double-ESC watcher (scripted keys + fake clock),
emit/run cancellation in WikiPipeline, per-file cancellation in
``ingest_documents``, the PipelineProgress watcher lifecycle, and the
top-level CLI exit handling.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from queue import Empty, SimpleQueue

import pytest

import kompend.cli.app as app_mod
from kompend.cli._esc_watch import DOUBLE_ESC_REASON, EscDoublePressWatcher
from kompend.cli._progress import PipelineProgress
from kompend.core.cancel import CANCEL_EXIT_CODE, CANCEL_WINDOW_SECONDS, CancelState, KompendCancelled
from kompend.core.config import resolve_config
from kompend.core.models import ParsedDocument
from kompend.core.paths import VaultPaths
from kompend.core.pipeline.config import PipelineConfig
from kompend.core.pipeline.ingest import ingest_documents
from kompend.core.pipeline.orchestrator import build_pipeline
from kompend.core.pipeline.stages._pipeline import WikiPipeline
from kompend.core.pipeline.state import TierStateStore
from kompend.preprocessing.parsers.base import BaseParser, FileExtension, ParserRouter

ESC = "\x1b"


# ── CancelState ──────────────────────────────────────────────────────────


class TestCancelState:
    def test_not_requested_initially(self) -> None:
        assert not CancelState().requested()
        assert CancelState().reason == ""

    def test_request_sets_requested(self) -> None:
        state = CancelState()
        state.request()
        assert state.requested()

    def test_first_reason_wins(self) -> None:
        state = CancelState()
        state.request("first")
        state.request("second")
        assert state.reason == "first"

    def test_default_reason(self) -> None:
        state = CancelState()
        state.request()
        assert state.reason == "cancelled by user"
        state2 = CancelState()
        state2.request("")
        assert state2.reason == "cancelled by user"

    def test_raise_if_requested_noop_when_not_requested(self) -> None:
        CancelState().raise_if_requested()  # must not raise

    def test_raise_if_requested_raises_with_reason(self) -> None:
        state = CancelState()
        state.request("double-ESC")
        with pytest.raises(KompendCancelled, match="double-ESC"):
            state.raise_if_requested()

    def test_kompend_cancelled_defaults(self) -> None:
        exc = KompendCancelled()
        assert exc.reason == "cancelled by user"
        assert str(exc) == "cancelled by user"

    def test_kompend_cancelled_bypasses_except_exception(self) -> None:
        assert issubclass(KompendCancelled, BaseException)
        assert not issubclass(KompendCancelled, Exception)

        def _swallower() -> None:
            try:
                raise KompendCancelled("stop")
            except Exception:  # noqa: BLE001 — proves the point
                msg = "must not be caught"
                raise RuntimeError(msg) from None

        with pytest.raises(KompendCancelled):
            _swallower()


# ── EscDoublePressWatcher ────────────────────────────────────────────────


class _ScriptedKeys:
    """Key reader fed from a queue; signals when each key is consumed."""

    def __init__(self) -> None:
        self.queue: SimpleQueue[str] = SimpleQueue()
        self.consumed = threading.Event()

    def __call__(self) -> str | None:
        try:
            key = self.queue.get_nowait()
        except Empty:
            return None
        self.consumed.set()
        return key

    def put(self, key: str) -> None:
        self.queue.put(key)

    def wait_consumed(self, timeout: float = 2.0) -> bool:
        got = self.consumed.wait(timeout)
        self.consumed.clear()
        return got


class _FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


class TestEscDoublePressWatcher:
    def test_double_esc_within_window_cancels(self) -> None:
        state = CancelState()
        keys = _ScriptedKeys()
        watcher = _make_watcher(state, keys=keys)
        assert watcher.start() is True
        keys.put(ESC)
        keys.put(ESC)
        try:
            assert _wait_until(state.requested), "cancel was not requested"
        finally:
            watcher.stop()
        assert state.reason == DOUBLE_ESC_REASON

    def test_single_esc_does_not_cancel(self) -> None:
        state = CancelState()
        keys = _ScriptedKeys()
        watcher = _make_watcher(state, keys=keys)
        watcher.start()
        keys.put(ESC)
        assert keys.wait_consumed()
        time.sleep(0.05)
        assert not state.requested()
        watcher.stop()

    def test_two_escs_apart_do_not_cancel(self) -> None:
        state = CancelState()
        keys = _ScriptedKeys()
        clock = _FakeClock()
        watcher = _make_watcher(state, clock=clock, keys=keys)
        watcher.start()
        keys.put(ESC)
        assert keys.wait_consumed()
        clock.advance(CANCEL_WINDOW_SECONDS + 0.1)
        keys.put(ESC)
        assert keys.wait_consumed()
        time.sleep(0.05)
        assert not state.requested()
        watcher.stop()

    def test_interleaved_key_resets_sequence(self) -> None:
        state = CancelState()
        keys = _ScriptedKeys()
        watcher = _make_watcher(state, keys=keys)
        watcher.start()
        keys.put(ESC)
        assert keys.wait_consumed()
        keys.put("a")
        assert keys.wait_consumed()
        keys.put(ESC)
        assert keys.wait_consumed()
        time.sleep(0.05)
        assert not state.requested()
        watcher.stop()

    def test_on_cancel_notified_with_reason(self) -> None:
        state = CancelState()
        keys = _ScriptedKeys()
        notices: list[str] = []
        watcher = _make_watcher(state, keys=keys, on_cancel=notices.append)
        watcher.start()
        keys.put(ESC)
        keys.put(ESC)
        try:
            assert _wait_until(lambda: bool(notices))
        finally:
            watcher.stop()
        assert notices == [DOUBLE_ESC_REASON]

    def test_non_tty_start_returns_false(self) -> None:
        state = CancelState()
        watcher = _make_watcher(state, is_tty=lambda: False)
        assert watcher.start() is False
        assert watcher._thread is None  # noqa: SLF001 — asserting no thread spawned

    def test_start_is_idempotent(self) -> None:
        state = CancelState()
        keys = _ScriptedKeys()
        watcher = _make_watcher(state, keys=keys)
        assert watcher.start() is True
        thread = watcher._thread  # noqa: SLF001 — identity check
        assert thread is not None
        assert watcher.start() is True
        assert watcher._thread is thread  # noqa: SLF001
        watcher.stop()

    def test_stop_joins_thread(self) -> None:
        state = CancelState()
        keys = _ScriptedKeys()
        watcher = _make_watcher(state, keys=keys)
        watcher.start()
        thread = watcher._thread  # noqa: SLF001
        watcher.stop()
        assert thread is not None
        assert not thread.is_alive()


class TestPipelineEmitCancel:
    def test_emit_raises_when_cancelled(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        state = CancelState()
        pipeline.cancel = state
        calls: list[tuple[str, int, int, str]] = []
        pipeline._progress_callback = lambda *args: calls.append(args)  # noqa: SLF001 — test hook
        state.request("double-ESC")
        with pytest.raises(KompendCancelled, match="double-ESC"):
            pipeline.emit("parse", 1, 2, "x.md")  # type: ignore[arg-type]
        assert calls == []

    def test_emit_dispatches_without_cancel(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        calls: list[tuple[str, int, int, str]] = []
        pipeline._progress_callback = lambda *args: calls.append(args)  # noqa: SLF001 — test hook
        pipeline.emit("parse", 1, 2, "x.md")  # type: ignore[arg-type]
        assert len(calls) == 1


class TestPipelineRunCancel:
    def test_run_saves_state_and_reraises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pipeline = _make_pipeline(tmp_path)
        pipeline.cancel = CancelState()
        pipeline.cancel.request("double-ESC")
        saved: list[bool] = []
        monkeypatch.setattr(pipeline, "save_state", lambda: saved.append(True))
        monkeypatch.setattr(pipeline, "_validate_prerequisites", lambda: None)
        monkeypatch.setattr(
            "kompend.core.pipeline.stages._pipeline.detect_resume_tier",
            lambda _pipeline: None,
        )
        with pytest.raises(KompendCancelled, match="double-ESC"):
            pipeline.run()
        assert saved == [True]


# ── ingest_documents cancellation ────────────────────────────────────────


class _FakeParser(BaseParser):
    """Claims ``.md`` sources and returns empty documents (no elements)."""

    supported_extensions = {FileExtension.MD}

    def parse(self, source_file: Path) -> ParsedDocument:
        return ParsedDocument(source_file=source_file, elements=[])


class _FakeRouter(ParserRouter):
    """Real router subclass that records every routed path."""

    def __init__(self) -> None:
        super().__init__()
        self.routed: list[Path] = []
        self.register(_FakeParser())

    def route(self, source_file: Path) -> BaseParser | None:
        self.routed.append(source_file)
        return super().route(source_file)


class TestIngestCancel:
    def test_stops_at_file_boundary_and_saves_state(self, tmp_path: Path) -> None:
        files = self._make_files(tmp_path)
        router = _FakeRouter()
        state = CancelState()

        def progress(current: int, total: int, detail: str) -> None:
            if current == 0:
                state.request("double-ESC")

        with pytest.raises(KompendCancelled, match="double-ESC"):
            ingest_documents(files, tmp_path, router, progress, cancel=state)
        # First file was fully processed; the second never started.
        assert len(router.routed) == 1
        # Tier records for completed files were persisted before the raise
        # (graceful-cancel contract).
        reloaded = TierStateStore(VaultPaths(tmp_path).compile_state_file).load()
        assert len(reloaded.entries) == 1

    def test_completes_without_cancel(self, tmp_path: Path) -> None:
        files = self._make_files(tmp_path)
        router = _FakeRouter()
        result = ingest_documents(files, tmp_path, router, None)
        assert result.count == 3
        assert len(router.routed) == 3

    @staticmethod
    def _make_files(tmp_path: Path, n: int = 3) -> list[Path]:
        files = []
        for i in range(n):
            f = tmp_path / "src" / f"doc{i}.md"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(f"# doc {i}\n", encoding="utf-8")
            files.append(f)
        return files


# ── PipelineProgress watcher lifecycle ───────────────────────────────────


class TestProgressWatcherWiring:
    def test_watcher_started_and_stopped_with_cancel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        started: list[bool] = []
        stopped: list[bool] = []
        monkeypatch.setattr(
            "kompend.cli._progress.EscDoublePressWatcher",
            self._fake_watcher_class(started, stopped),
        )
        tracker = PipelineProgress(enabled=True, _file_tracker=None, cancel=CancelState())
        tracker.start()
        assert started == [True]
        tracker.stop()
        assert stopped == [True]

    def test_no_watcher_without_cancel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        started: list[bool] = []
        stopped: list[bool] = []
        monkeypatch.setattr(
            "kompend.cli._progress.EscDoublePressWatcher",
            self._fake_watcher_class(started, stopped),
        )
        tracker = PipelineProgress(enabled=True, _file_tracker=None)
        tracker.start()
        tracker.stop()
        assert started == []
        assert stopped == []

    def test_disabled_tracker_never_starts_watcher(self, monkeypatch: pytest.MonkeyPatch) -> None:
        started: list[bool] = []
        stopped: list[bool] = []
        monkeypatch.setattr(
            "kompend.cli._progress.EscDoublePressWatcher",
            self._fake_watcher_class(started, stopped),
        )
        tracker = PipelineProgress(enabled=False, cancel=CancelState())
        tracker.start()
        tracker.finish()
        assert started == []

    @staticmethod
    def _fake_watcher_class(
        started: list[bool],
        stopped: list[bool],
    ) -> type:
        class _FakeWatcher:
            def __init__(self, state: CancelState, *, on_cancel: object = None) -> None:
                self.state = state

            def start(self) -> bool:
                started.append(True)
                return True

            def stop(self) -> None:
                stopped.append(True)

        return _FakeWatcher


# ── Top-level CLI handling ───────────────────────────────────────────────


class TestAppCancelledExit:
    def test_cancelled_exits_130_with_runlog(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(app_mod, "_configure_logfire", lambda: None)
        monkeypatch.setattr(app_mod, "configure_logging", lambda: None)
        monkeypatch.setattr(app_mod, "_configure_streams_utf8", lambda: None)
        monkeypatch.setattr(app_mod, "_extract_handle", lambda: None)

        recorded: dict[str, object] = {}
        monkeypatch.setattr(
            app_mod,
            "log_run_end",
            lambda handle, **kwargs: recorded.update(kwargs),
        )

        def _raise_cancelled() -> None:
            raise KompendCancelled("double-ESC")

        monkeypatch.setattr(app_mod, "app", _raise_cancelled)

        with pytest.raises(SystemExit) as excinfo:
            app_mod.main()

        assert excinfo.value.code == CANCEL_EXIT_CODE
        assert recorded.get("status") == "cancelled"
        assert recorded.get("exit_code") == CANCEL_EXIT_CODE
        assert recorded.get("error") == "double-ESC"
        err = capsys.readouterr().err
        assert "Cancelled (double-ESC)" in err


def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _make_watcher(
    state: CancelState,
    *,
    clock: _FakeClock | None = None,
    keys: _ScriptedKeys | None = None,
    is_tty: Callable[[], bool] | None = None,
    on_cancel: Callable[[str], None] | None = None,
) -> EscDoublePressWatcher:
    return EscDoublePressWatcher(
        state,
        poll_interval=0.005,
        read_key=keys,
        is_tty=is_tty or (lambda: True),
        clock=clock or _FakeClock(),
        on_cancel=on_cancel,
    )


# ── WikiPipeline emit/run cancellation ───────────────────────────────────


def _make_pipeline(tmp_path: Path) -> WikiPipeline:
    cfg = resolve_config(base_dir=tmp_path)
    config = PipelineConfig.from_resolved(
        cfg,
        source_dir=tmp_path / "sources",
        output_dir=tmp_path / "wiki",
        input_files=None,
    )
    return WikiPipeline(config=config)


def test_build_pipeline_accepts_cancel(tmp_path: Path) -> None:
    cfg = resolve_config(base_dir=tmp_path)
    config = PipelineConfig.from_resolved(
        cfg,
        source_dir=tmp_path / "sources",
        output_dir=tmp_path / "wiki",
        input_files=None,
    )
    state = CancelState()
    pipeline = build_pipeline(config, cancel=state)
    assert pipeline.cancel is state
