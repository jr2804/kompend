"""Tests for CLI progress bars (_progress.py)."""

from __future__ import annotations

from knox.cli._progress import PipelineProgress, _should_show_progress


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
