"""Tests for CLI progress bars (_progress.py) and progress callbacks."""

from __future__ import annotations

from kompend.cli._pipeline_helpers import make_progress_callback
from kompend.cli._progress import PipelineProgress, _should_show_progress


class TestShowProgress:
    def test_force_true(self) -> None:
        assert _should_show_progress(force=True) is True

    def test_force_false(self) -> None:
        assert _should_show_progress(force=False) is False

    def test_auto_non_tty(self) -> None:
        # In tests, stdout is not a TTY
        assert _should_show_progress(force=None) is False


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

    Events emitted by ``parse_and_chunk`` (core/pipeline/stages/_parse.py):
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
