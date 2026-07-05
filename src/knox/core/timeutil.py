"""Local-timezone helpers for timestamps.

All user-facing and logged timestamps in KNOX use the local system timezone
(read from the OS), not UTC. Centralise that here so every call site is
consistent and the timezone source is a single knob.
"""

from __future__ import annotations

from datetime import datetime


def now_local() -> datetime:
    """Current time in the local system timezone (timezone-aware).

    ``datetime.now().astimezone()`` attaches the local tz to an otherwise
    naive local time, yielding e.g. ``2026-07-05T15:23:01+02:00``.
    """
    return datetime.now().astimezone()


def iso_local(*, timespec: str = "seconds") -> str:
    """ISO 8601 timestamp in the local timezone, e.g.
    ``2026-07-05T15:23:01+02:00``.

    Use for machine-readable fields (frontmatter, JSONL, structured logs).
    """
    return now_local().isoformat(timespec=timespec)


def display_local(fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Format the current local time for human display, e.g.
    ``2026-07-05 15:23``.
    """
    return now_local().strftime(fmt)
