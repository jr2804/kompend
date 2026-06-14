"""Rich progress bars for KNOX CLI operations.

Provides two levels of progress display:
1. **Across-file progress** — outer bar showing file X of N with per-file status
2. **Within-file progress** — inner pipeline steps for the current file

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

    def step(self, name: str, status: str = "") -> None:
        pass

    def finish_file(self, file_name: str, status: str, duration: float = 0) -> None:
        pass

    def error(self, file_name: str, message: str) -> None:
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


# ── Step tracker (within-file) ──────────────────────────────────────────


PIPELINE_STEPS = ["Download", "Parse", "Chunk", "Resolve", "Compile", "Write"]


@dataclass
class StepTracker:
    """Tracks within-file pipeline steps.

    Usage::

        steps = StepTracker()
        steps.start("paper.pdf")
        steps.step("Parse", "OK")
        steps.step("Chunk", "OK")
        steps.step("Resolve", "running...")
        steps.finish("paper.pdf", "OK")
    """

    parent_progress: Progress | None = None
    _task: TaskID | None = None
    _current_step: int = 0
    _total_steps: int = 6
    _step_statuses: list[str] = field(default_factory=list)

    def start(self, file_name: str) -> None:
        """Create a sub-task for the current file's progress."""
        self._current_step = 0
        self._step_statuses = []
        if self.parent_progress:
            self._task = self.parent_progress.add_task(
                f"  {file_name}",
                total=self._total_steps,
            )

    def step(self, name: str, status: str = "") -> None:
        """Advance to the next pipeline step."""
        self._current_step += 1
        self._step_statuses.append(f"✓ {name}")
        if self.parent_progress and self._task is not None:
            self.parent_progress.update(
                self._task,
                advance=1,
                description=f"  {name}",
            )

    def finish(self, file_name: str, status: str) -> None:
        """Mark the current file's sub-task as complete."""
        if self.parent_progress and self._task is not None:
            self.parent_progress.update(
                self._task,
                completed=self._total_steps,
                description=f"  ✓ {file_name} ({status})",
            )


# ── Combined tracker ────────────────────────────────────────────────────


@dataclass
class PipelineProgress:
    """Combined across-file and within-file progress display.

    This is the main entry point for progress tracking in the pipeline.
    """

    enabled: bool = True
    _file_tracker: FileProgressTracker | None = None
    _step_tracker: StepTracker | None = None
    _file_index: int = 0

    def start(self) -> None:
        """Start the overall progress display."""
        if self._file_tracker:
            self._file_tracker.start()
        # Connect step tracker to the now-created Rich Progress instance.
        if self._step_tracker and self._file_tracker:
            self._step_tracker.parent_progress = self._file_tracker._progress

    def start_file(self, file_name: str) -> None:
        """Start tracking progress for a new file."""
        self._file_index += 1
        if self._step_tracker:
            self._step_tracker.start(file_name)

    def step(self, name: str, status: str = "") -> None:
        """Advance one pipeline step for the current file."""
        if self._step_tracker:
            self._step_tracker.step(name, status)

    def finish_file(self, file_name: str, status: str = "OK") -> None:
        """Mark the current file as complete."""
        if self._step_tracker:
            self._step_tracker.finish(file_name, status)
        if self._file_tracker:
            self._file_tracker.advance()

    def message(self, msg: str) -> None:
        """Update the overall task description with a coarse status message.

        Used for non-file phases (resolve, compile, output) where
        per-file tracking doesn't apply.
        """
        if self._file_tracker and self._file_tracker._progress and self._file_tracker._overall_task is not None:
            self._file_tracker._progress.update(
                self._file_tracker._overall_task,
                description=msg,
            )

    def finish(self) -> None:
        """Mark overall progress as complete."""
        if self._file_tracker:
            self._file_tracker.complete()
            self._file_tracker.stop()

    def stop(self) -> None:
        """Stop the progress display."""
        if self._file_tracker:
            self._file_tracker.stop()

    def error(self, file_name: str, message: str) -> None:
        """Log an error for a file (shown in task list)."""
        pass  # errors are logged via finish_file

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
        # StepTracker parent_progress is set lazily in start() after
        # FileProgressTracker creates the Rich Progress widget.
        step_tracker = StepTracker(parent_progress=None)
        return cls(enabled=True, _file_tracker=file_tracker, _step_tracker=step_tracker)


@contextmanager
def null_progress() -> Generator[NullProgressTracker, Any]:
    """Context manager producing a no-op progress tracker.

    Use this in non-TTY or quiet mode instead of a real progress bar.
    """
    yield NullProgressTracker()
