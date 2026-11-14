"""Tests for CLI progress bars (_progress.py) and progress callbacks."""

from __future__ import annotations

import io
import sys
from collections.abc import Iterator

import pytest

from kompend.cli.runtime.progress import PipelineProgress, _make_progress_console, _should_show_progress
from kompend.cli.workflow.pipeline import make_progress_callback
from kompend.logging import get_err_console


class _Tty(io.StringIO):
    """StringIO that reports itself as a terminal."""

    def isatty(self) -> bool:
        return True


class TestShowProgress:
    def test_force_true(self) -> None:
        assert _should_show_progress(force=True) is True

    def test_force_false(self) -> None:
        assert _should_show_progress(force=False) is False

    def test_auto_non_tty(self) -> None:
        # In tests, stdout is not a TTY
        assert _should_show_progress(force=None) is False

    @pytest.mark.usefixtures("_fresh_err_console_cache")
    def test_auto_either_stream_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The bar keys on the console's own stream, with a TTY stdout as backup.

        Windows/PowerShell case: stdout is a TTY but stderr is not reported
        as one, yet the console is visible — that host is interactive.
        """
        monkeypatch.setattr(sys, "stdout", _Tty())
        assert _should_show_progress(force=None) is True

        monkeypatch.setattr(sys, "stderr", _Tty())
        assert _should_show_progress(force=None) is True


class TestMakeProgressConsole:
    @pytest.mark.usefixtures("_fresh_err_console_cache")
    def test_stdout_tty_stderr_not_forces_terminal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression (knox-hoga): stdout TTY + stderr not a TTY in auto mode.

        The forced console makes Rich's live refresh thread run, so the
        elapsed timer and phase label keep moving instead of freezing the
        first frame — a frozen bar is indistinguishable from a hang.
        """
        monkeypatch.setattr(sys, "stdout", _Tty())
        # stderr stays the pytest-captured (non-TTY) stream.
        assert sys.stderr.isatty() is False

        pp = PipelineProgress.create(total_files=2, force_progress=None)
        assert pp.enabled is True
        assert pp._file_tracker is not None
        assert pp._file_tracker.console.is_terminal is True

    @pytest.mark.usefixtures("_fresh_err_console_cache")
    def test_stderr_tty_reuses_cached_console(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Genuinely TTY stderr keeps the shared console (RichHandler interleaving unchanged)."""
        monkeypatch.setattr(sys, "stderr", _Tty())

        assert _make_progress_console(force=False) is get_err_console()

    @pytest.mark.usefixtures("_fresh_err_console_cache")
    def test_stderr_redirected_no_bar(self) -> None:
        """Piped/CI output (neither stream a TTY) stays bar-free."""
        assert _should_show_progress(force=None) is False
        pp = PipelineProgress.create(total_files=2, force_progress=None)
        assert pp.enabled is False


class TestPipelineProgress:
    def test_create_disabled(self) -> None:
        pp = PipelineProgress.create(total_files=5, force_progress=False)
        assert pp.enabled is False

    def test_create_enabled(self) -> None:
        pp = PipelineProgress.create(total_files=5, force_progress=True)
        assert pp.enabled is True

    def test_lifecycle(self) -> None:
        """Parse phase: start → update_progress → finish."""
        pp = PipelineProgress.create(total_files=2, force_progress=True)
        assert pp.enabled is True
        pp.start()
        pp.update_progress(1, "[1/2] doc1.pdf")
        pp.update_progress(2, "[2/2] doc2.pdf")
        pp.reset_for_phase(120, "Resolving chunks (120 total)…")
        pp.update_progress(60, "Resolving [60/120]")
        pp.update_progress(120, "Resolve complete")
        pp.reset_for_phase(None, "Compiling wiki…")
        pp.message("Compile complete")
        pp.finish()
        pp.stop()


class _RecordingTracker(PipelineProgress):
    """Test double that records handler calls (Rich internals unused)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object]] = []

    def start(self) -> None:
        self.calls.append(("start", None, None))

    def update_progress(self, current: int, description: str) -> None:
        self.calls.append(("update", current, description))

    def reset_for_phase(self, total: int | None, description: str) -> None:
        self.calls.append(("reset", total, description))

    def message(self, msg: str) -> None:
        self.calls.append(("message", None, msg))


class TestMakeProgressCallbackSummary:
    """The summary sub-phase: per-tree bar during LLM summarization.

    Events emitted by ``parse_and_chunk`` (core/pipeline/stages/parse.py):
    sentinel (0, N, "starting") → per-unit (i, N, label) → (N, N, "done").
    """

    def test_sentinel_resets_bar_with_total(self) -> None:
        tracker = _RecordingTracker()
        cb = make_progress_callback(tracker)
        cb("summary", 0, 233, "starting")
        assert tracker.calls == [("reset", 233, "Summarizing sections (233 source(s) to summarize)…")]

    def test_per_unit_updates_bar(self) -> None:
        tracker = _RecordingTracker()
        cb = make_progress_callback(tracker)
        cb("summary", 17, 233, "doc17.pdf")
        assert tracker.calls == [("update", 17, "[17/233] doc17.pdf")]

    def test_done_finishes_phase(self) -> None:
        tracker = _RecordingTracker()
        cb = make_progress_callback(tracker)
        cb("summary", 233, 233, "done")
        assert tracker.calls == [("update", 233, "Summarized 233 source(s)")]

    def test_empty_done_is_message_not_bar(self) -> None:
        tracker = _RecordingTracker()
        cb = make_progress_callback(tracker)
        cb("summary", 0, 0, "starting")
        cb("summary", 0, 0, "done")
        assert tracker.calls == [
            ("reset", 0, "Summarizing sections (0 source(s) to summarize)…"),
            ("message", None, "No sources to summarize"),
        ]


@pytest.fixture
def _fresh_err_console_cache() -> Iterator[None]:
    """Clear the cached err console around tests that build consoles.

    :func:`get_err_console` is ``functools.cache``d and binds ``sys.stderr``
    at construction; tests that fake the streams must not leak a fake-bound
    (or stale real-bound) console into other tests.
    """
    get_err_console.cache_clear()
    yield
    get_err_console.cache_clear()
