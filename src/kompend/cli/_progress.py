"""Rich progress bars for kompend CLI operations.

Provides a single overall bar that is reset per pipeline phase (parse →
resolve → compile → output).  No per-file sub-tasks are created, keeping
the display clean and avoiding accumulation of stale tasks.

Auto-suppresses in non-TTY mode and respects ``--quiet`` / ``--no-progress``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column
from rich.text import Text

from kompend.logging import get_err_console


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
    console: Console = field(default_factory=get_err_console)
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
            TextColumn("[progress.description]{task.description}", table_column=Column(no_wrap=True)),
            BarColumn(bar_width=None),
            PercentageColumn(),
            EtaColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
            refresh_per_second=2,
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
            ft._progress.refresh()

    def reset_for_phase(self, total: int | None, description: str) -> None:
        """Reset the bar for a new pipeline phase.

        *total* sets the new bar total; ``None`` makes it indeterminate.

        Rich's ``Progress.update(total=None)`` treats ``None`` as a
        no-op sentinel ("don't change"), so it can *never* switch a task
        to indeterminate. To actually change the total we recreate the
        task via the proper ``add_task`` API.
        """
        ft = self._file_tracker
        if ft and ft._progress and ft._overall_task is not None:
            ft._progress.remove_task(ft._overall_task)
            ft._overall_task = ft._progress.add_task(description, total=total)
            ft._progress.refresh()

    def message(self, msg: str) -> None:
        """Update the bar description without changing progress."""
        ft = self._file_tracker
        if ft and ft._progress and ft._overall_task is not None:
            ft._progress.update(ft._overall_task, description=msg)
            ft._progress.refresh()

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
        """Create a PipelineProgress, auto-detecting TTY.

        When progress is forced (``--progress``), the bar console is created
        with ``force_terminal=True`` so Rich's live refresh thread runs even
        when TTY auto-detection fails (e.g. some Windows/PowerShell hosts).
        Without this, Rich renders the bar once and never refreshes — the bar
        freezes mid-phase while events keep firing underneath.
        """
        show = _should_show_progress(force_progress)
        if not show:
            return cls(enabled=False)

        console = _make_progress_console(force=bool(force_progress))
        file_tracker = FileProgressTracker(total=total_files, console=console)
        return cls(enabled=True, _file_tracker=file_tracker)


# ── Rich progress tracker ───────────────────────────────────────────────


class EtaColumn(ProgressColumn):
    """ETA column using direct elapsed/completed math instead of Rich's speed estimation.

    Rich's built-in ``TimeRemainingColumn`` relies on an internal ``_progress``
    deque that can fall back to ``-:--:--`` when speed samples are sparse or
    the task is recreated between phases.  This column calculates remaining
    time directly from ``completed / time_elapsed``, which is always
    well-defined once at least one item is done.
    """

    def render(self, task: Task) -> Text:  # noqa: PLR6301  # mandatory Rich ProgressColumn API
        if task.total is None or task.completed <= 0:
            return Text("-:--:--", style="progress.remaining")
        elapsed = task.elapsed
        if not elapsed or elapsed <= 0:
            return Text("-:--:--", style="progress.remaining")
        speed = task.completed / elapsed
        remaining = (task.total - task.completed) / speed
        if remaining <= 0:
            return Text("-:--:--", style="progress.remaining")
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        seconds = int(remaining % 60)
        if hours:
            return Text(f"{hours}:{minutes:02d}:{seconds:02d}", style="progress.remaining")
        return Text(f"{minutes}:{seconds:02d}", style="progress.remaining")


class PercentageColumn(ProgressColumn):
    """Percentage column that hides for indeterminate (total=None) tasks.

    Rich renders ``0%`` for indeterminate tasks, which reads as "stuck".
    Return an empty cell instead — the spinner + elapsed time + sub-step
    description convey activity for indeterminate phases (compile/output).
    """

    def render(self, task: Task) -> Text:  # noqa: PLR6301  # mandatory Rich ProgressColumn API
        if task.total is None:
            return Text("")
        pct = task.percentage or 0.0
        return Text(f"{pct:>3.0f}%", style="progress.percentage")


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


def _make_progress_console(force: bool) -> Console:
    """Build the stderr console that owns the progress bar.

    When *force* is set (``--progress`` passed explicitly) the console is
    forced into terminal mode so Rich's live refresh thread runs. Without
    this, environments where ``sys.stderr.isatty()`` is unreliable (some
    Windows/PowerShell hosts) get a single static render that never updates.

    When *force* is unset (auto/TTY mode) the shared cached err console is
    reused so log output via RichHandler stays Live-aware (prints above the
    bar) instead of clobbering it.
    """
    if not force:
        return get_err_console()
    return Console(
        stderr=True,
        force_terminal=True,
        no_color=not sys.stderr.isatty(),
    )
