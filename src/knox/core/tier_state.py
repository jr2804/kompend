"""Per-source tier state persistence for Q20 tiered compilation.

Tracks the highest tier that has been completed for each source file,
its content hash at the time of compilation, and a few auxiliary
counters. The state lives in ``.knox/compile_state.json`` inside the
output directory so a restart picks up exactly where the previous run
left off.

The store is deliberately simple: a JSON file keyed by source-file
stem (and resolved path) — the kind of state that has to survive a
process restart and not require sqlite migrations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SourceState:
    """Persistent state for a single source file."""

    stem: str
    source_path: str
    content_hash: str
    current_tier: int = 0
    last_compile: str = ""
    search_hits: int = 0
    last_hit: str = ""
    ingested_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceState:
        return cls(
            stem=data.get("stem", ""),
            source_path=data.get("source_path", ""),
            content_hash=data.get("content_hash", ""),
            current_tier=int(data.get("current_tier", 0)),
            last_compile=data.get("last_compile", ""),
            search_hits=int(data.get("search_hits", 0)),
            last_hit=data.get("last_hit", ""),
            ingested_by=data.get("ingested_by", ""),
        )


@dataclass
class TierState:
    """Aggregate state for all sources."""

    entries: dict[str, SourceState] = field(default_factory=dict)

    def tier_distribution(self) -> dict[int, int]:
        """Return a count map from tier number to source count."""
        out: dict[int, int] = {}
        for e in self.entries.values():
            out[e.current_tier] = out.get(e.current_tier, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"entries": {k: v.to_dict() for k, v in self.entries.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TierState:
        entries = {k: SourceState.from_dict(v) for k, v in data.get("entries", {}).items()}
        return cls(entries=entries)


class TierStateStore:
    """JSON-backed store for per-source tier state.

    Parameters
    ----------
    state_path:
        File to read/write. Created lazily on first save.
    """

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path

    def load(self) -> TierState:
        """Read the state file, returning an empty state if missing or invalid."""
        if not self.state_path.exists():
            return TierState()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.debug("Tier state file unreadable — starting fresh", exc_info=True)
            return TierState()
        if not isinstance(data, dict):
            return TierState()
        return TierState.from_dict(data)

    def save(self, state: TierState) -> None:
        """Persist the state to disk. Creates parent directories as needed."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def record(
        self,
        state: TierState,
        stem: str,
        source_path: Path,
        content_hash: str,
        tier: int,
        ingested_by: str | None = None,
    ) -> SourceState:
        """Insert or update the entry for ``stem`` and return it.

        When *ingested_by* is ``None`` (the compile path) the existing
        value is preserved; any string (including ``""``) overwrites it.
        """
        _ = self  # noqa: PLR6301 — kept as instance method for future per-store config
        now = datetime.now(UTC).isoformat(timespec="seconds")
        existing = state.entries.get(stem)
        if existing is None:
            entry = SourceState(
                stem=stem,
                source_path=str(source_path),
                content_hash=content_hash,
                current_tier=tier,
                last_compile=now,
                ingested_by=ingested_by or "",
            )
        else:
            entry = SourceState(
                stem=stem,
                source_path=str(source_path),
                content_hash=content_hash,
                current_tier=tier,
                last_compile=now,
                search_hits=existing.search_hits,
                last_hit=existing.last_hit,
                ingested_by=ingested_by if ingested_by is not None else existing.ingested_by,
            )
        state.entries[stem] = entry
        return entry

    def record_hit(self, state: TierState, stem: str) -> None:
        """Increment search-hit counter for a source."""
        _ = self  # noqa: PLR6301 — instance method for future per-store config
        existing = state.entries.get(stem)
        if existing is None:
            return
        existing.search_hits += 1
        existing.last_hit = datetime.now(UTC).isoformat(timespec="seconds")
        state.entries[stem] = existing


def auto_promote_demote(
    state: TierState,
    *,
    promote_threshold: int = 3,
    demote_days: int = 90,
    now: datetime | None = None,
) -> int:
    """Apply auto-promote / auto-demote rules in place.

    Returns the number of state entries that changed tier.
    """
    now = now or datetime.now(UTC)
    changed = 0
    for entry in state.entries.values():
        new_tier = entry.current_tier
        if entry.search_hits >= promote_threshold and entry.current_tier < 3:
            new_tier = 3
        elif entry.last_compile:
            try:
                last_dt = datetime.fromisoformat(entry.last_compile)
            except ValueError:
                last_dt = now
            age_days = (now - last_dt).days
            if age_days >= demote_days and entry.current_tier > 1:
                new_tier = 1
        if new_tier != entry.current_tier:
            entry.current_tier = new_tier
            state.entries[entry.stem] = entry
            changed += 1
    return changed
