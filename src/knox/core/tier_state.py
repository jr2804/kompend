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
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from knox.core.models import MetadataField
from knox.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SourceState:
    """Persistent state for a single source file."""

    stem: str
    source_file: str
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
            source_file=data.get(MetadataField.SOURCE_FILE) or data.get(MetadataField.SOURCE_PATH, ""),
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
    state_file:
        File to read/write. Created lazily on first save.
    """

    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file

    def load(self) -> TierState:
        """Read the state file.

        Returns an empty state when the file is missing (fresh start). A
        corrupt file is a bug — it raises so the real cause surfaces instead
        of silently wiping tier progress.
        """
        if not self.state_file.exists():
            return TierState()
        data = json.loads(self.state_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return TierState()
        return TierState.from_dict(data)

    def save(self, state: TierState) -> None:
        """Persist the state to disk. Creates parent directories as needed."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(state.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def record(
        self,
        state: TierState,
        stem: str,
        source_file: Path,
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
                source_file=str(source_file),
                content_hash=content_hash,
                current_tier=tier,
                last_compile=now,
                ingested_by=ingested_by or "",
            )
        else:
            entry = SourceState(
                stem=stem,
                source_file=str(source_file),
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
