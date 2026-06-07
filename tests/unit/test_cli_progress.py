"""Tests for CLI progress bars (_progress.py)."""

from __future__ import annotations

from knox.cli._progress import NullProgressTracker, PipelineProgress, _should_show_progress


class TestShowProgress:
    def test_force_true(self) -> None:
        assert _should_show_progress(force=True) is True

    def test_force_false(self) -> None:
        assert _should_show_progress(force=False) is False

    def test_auto_non_tty(self) -> None:
        # In tests, stdout is not a TTY
        assert _should_show_progress(force=None) is False


class TestNullProgressTracker:
    def test_all_methods_noop(self) -> None:
        tracker = NullProgressTracker()
        # None of these should raise
        tracker.start()
        tracker.advance()
        tracker.complete()
        tracker.step("Parse")
        tracker.finish_file("test.pdf", "OK")
        tracker.error("test.pdf", "message")
        tracker.stop()


class TestPipelineProgress:
    def test_create_disabled(self) -> None:
        pp = PipelineProgress.create(total_files=5, force_progress=False)
        assert pp.enabled is False

    def test_create_enabled(self) -> None:
        pp = PipelineProgress.create(total_files=5, force_progress=True)
        assert pp.enabled is True

    def test_lifecycle(self) -> None:
        pp = PipelineProgress.create(total_files=2, force_progress=True)
        assert pp.enabled is True
        pp.start()
        pp.start_file("doc1.pdf")
        pp.step("Parse", "OK")
        pp.step("Chunk", "OK")
        pp.finish_file("doc1.pdf")
        pp.start_file("doc2.pdf")
        pp.step("Parse", "OK")
        pp.finish_file("doc2.pdf")
        pp.complete = lambda: None  # avoid actual stop
        pp.finish()
        pp.stop()
