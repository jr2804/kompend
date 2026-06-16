"""Rich progress bars for KNOX CLI operations.

Provides a single overall bar that is reset per pipeline phase (parse →
resolve → compile → output).  No per-file sub-tasks are created, keeping
the display clean and avoiding accumulation of stale tasks.

Auto-suppresses in non-TTY mode and respects ``--quiet`` / ``--no-progress``.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# ── Helpers ─────────────────────────────────────────────────────────────


def _should_show_progress(force: bool | None = None) -> bool:
    """Determine if progress display should be enabled.

    Returns True if:
    - ``force`` is True
    - ``force`` is None and stdout is a TTY
    """
    if force is True:
        return True
    if force is False:
        return False
    return sys.stdout.isatty()


# ── Null progress (non-TTY / quiet mode) ────────────────────────────────


@dataclass
class NullProgressTracker:
    """No-op progress tracker for non-TTY or quiet mode."""

    def start(self) -> None:
        pass

    def advance(self, n: int = 1) -> None:
        pass

    def complete(self) -> None:
        pass

    def update_progress(self, current: int, description: str) -> None:
        pass

    def reset_for_phase(self, total: int | None, description: str) -> None:
        pass

    def message(self, msg: str) -> None:
        pass

    def stop(self) -> None:
        pass


# ── Rich progress tracker ───────────────────────────────────────────────


@dataclass
class FileProgressTracker:
    """Manages across-file progress display.

    Usage::

        tracker = FileProgressTracker(total=10)
        tracker.start()
        for file in files:
            tracker.advance(file.name)
            # ... process file ...
            tracker.finish_file(file.name, "OK", elapsed)
        tracker.stop()
    """

    total: int
    console: Console = field(default_factory=lambda: Console(stderr=True))
    _progress: Progress | None = None
    _overall_task: TaskID | None = None
    _start_time: float = 0.0
    _started: bool = False

    def start(self) -> None:
        """Create and start the progress display (idempotent)."""
        if self._started:
            return
        self._started = True
        self._start_time = time.time()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )
        self._progress.start()
        self._overall_task = self._progress.add_task(
            f"Processing files ({self.total} total)",
            total=self.total,
        )

    def advance(self, n: int = 1) -> None:
        """Advance the overall progress bar by *n* steps."""
        if self._progress and self._overall_task is not None:
            self._progress.update(self._overall_task, advance=n)

    def complete(self) -> None:
        """Mark the overall progress as complete (100%)."""
        if self._progress and self._overall_task is not None:
            self._progress.update(self._overall_task, completed=self.total)

    def stop(self) -> None:
        """Stop and clear the progress display."""
        if self._progress:
            self._progress.stop()
            self._progress = None


# ── Combined tracker ────────────────────────────────────────────────────


@dataclass
class PipelineProgress:
    """Single-bar progress display that resets per pipeline phase.

    The overall bar tracks file count during *parse*, chunk count during
    *resolve*, and switches to indeterminate mode for *compile* / *output*.
    """

    enabled: bool = True
    _file_tracker: FileProgressTracker | None = None

    def start(self) -> None:
        """Start the overall progress display."""
        if self._file_tracker:
            self._file_tracker.start()

    def update_progress(self, current: int, description: str) -> None:
        """Set the bar's completed count and description."""
        ft = self._file_tracker
        if ft and ft._progress and ft._overall_task is not None:
            ft._progress.update(ft._overall_task, completed=current, description=description)

    def reset_for_phase(self, total: int | None, description: str) -> None:
        """Reset the bar for a new pipeline phase.

        *total* sets the new bar total; ``None`` makes it indeterminate.
        """
        ft = self._file_tracker
        if ft and ft._progress and ft._overall_task is not None:
            ft._progress.update(
                ft._overall_task,
                total=total,
                completed=0,
                description=description,
            )

    def message(self, msg: str) -> None:
        """Update the bar description without changing progress."""
        ft = self._file_tracker
        if ft and ft._progress and ft._overall_task is not None:
            ft._progress.update(ft._overall_task, description=msg)

    def finish(self) -> None:
        """Mark progress complete and stop the display."""
        if self._file_tracker:
            self._file_tracker.complete()
            self._file_tracker.stop()

    def stop(self) -> None:
        """Stop the progress display."""
        if self._file_tracker:
            self._file_tracker.stop()

    @classmethod
    def create(
        cls,
        total_files: int,
        force_progress: bool | None = None,
    ) -> PipelineProgress:
        """Create a PipelineProgress, auto-detecting TTY."""
        show = _should_show_progress(force_progress)
        if not show:
            return cls(enabled=False)

        file_tracker = FileProgressTracker(total=total_files)
        return cls(enabled=True, _file_tracker=file_tracker)


@contextmanager
def null_progress() -> Generator[NullProgressTracker, Any]:
    """Context manager producing a no-op progress tracker.

    Use this in non-TTY or quiet mode instead of a real progress bar.
    """
    yield NullProgressTracker()
