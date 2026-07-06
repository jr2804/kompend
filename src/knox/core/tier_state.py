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
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from knox.core.models import MetadataField
from knox.core.pipeline_defs import TIER_LEVEL, Tier
from knox.core.timeutil import iso_local, now_local
from knox.logging import get_logger

logger = get_logger(__name__)

# Tier level at which a source is considered fully compiled (OUTPUT = 4).
_OUTPUT_TIER_LEVEL = TIER_LEVEL[Tier.OUTPUT]


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

    def __init__(self, state_file: Path, base_dir: Path | None = None) -> None:
        self.state_file = state_file
        self._base_dir = base_dir

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
        now = iso_local()
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
        existing.last_hit = iso_local()
        state.entries[stem] = existing


class CompileStatus(StrEnum):
    """High-level compile status for display in ``knox status``."""

    NOT_INITIALIZED = "not-initialized"
    NO_SOURCES = "no-sources"
    NOT_INDEXED = "not-indexed"
    IN_PROGRESS = "in-progress"
    PARTIALLY_COMPILED = "partially-compiled"
    COMPILED = "compiled"
    BROKEN = "broken"


@dataclass(frozen=True)
class CompileStatusInfo:
    """Rich compile status for ``knox status`` output."""

    status: CompileStatus
    total: int = 0
    indexed: int = 0
    compiled: int = 0
    error: str = ""

    def label(self) -> str:
        """Human-readable one-line label for CLI display."""
        s = self.status
        if s is CompileStatus.NOT_INITIALIZED:
            result = "not initialized"
        elif s is CompileStatus.NO_SOURCES:
            result = "initialized, no sources"
        elif s is CompileStatus.NOT_INDEXED:
            result = f"{self.total} source(s) not indexed"
        elif s is CompileStatus.IN_PROGRESS:
            result = f"{self.indexed} source(s) ingested, not compiled"
        elif s is CompileStatus.PARTIALLY_COMPILED:
            result = f"partially compiled ({self.compiled}/{self.total})"
        elif s is CompileStatus.COMPILED:
            result = f"compiled ({self.compiled} source(s))"
        elif s is CompileStatus.BROKEN:
            result = f"broken: {self.error}" if self.error else "broken"
        else:
            result = str(s)
        return result


def classify_compile_status(
    *,
    state_dir_exists: bool,
    sources_count: int,
    tier_state: TierState | None,
    wiki_exists: bool,
    error: str = "",
) -> CompileStatusInfo:
    """Classify compile state for human-friendly status display.

    Parameters
    ----------
    state_dir_exists:
        Whether the ``.knox/`` metadata directory exists on disk.
    sources_count:
        Number of source files found in the sources directory.
    tier_state:
        Loaded :class:`TierState`, or ``None`` if the state file is
        missing or could not be parsed.
    wiki_exists:
        Whether the compiled wiki output file (``wiki.md``) exists.

    Error:
        Error message when the tier state file is corrupt.
    """
    if error:
        info = CompileStatusInfo(status=CompileStatus.BROKEN, total=sources_count, error=error)
    elif not state_dir_exists:
        info = CompileStatusInfo(status=CompileStatus.NOT_INITIALIZED)
    elif sources_count == 0:
        info = CompileStatusInfo(status=CompileStatus.NO_SOURCES)
    elif tier_state is None or not tier_state.entries:
        info = CompileStatusInfo(status=CompileStatus.NOT_INDEXED, total=sources_count)
    else:
        dist = tier_state.tier_distribution()
        indexed = len(tier_state.entries)
        compiled = sum(c for tier, c in dist.items() if tier >= _OUTPUT_TIER_LEVEL)
        if compiled >= indexed and compiled > 0:
            info = CompileStatusInfo(
                status=CompileStatus.COMPILED,
                total=sources_count,
                indexed=indexed,
                compiled=compiled,
            )
        elif compiled > 0 or wiki_exists:
            info = CompileStatusInfo(
                status=CompileStatus.PARTIALLY_COMPILED,
                total=sources_count,
                indexed=indexed,
                compiled=compiled,
            )
        else:
            info = CompileStatusInfo(
                status=CompileStatus.IN_PROGRESS,
                total=sources_count,
                indexed=indexed,
                compiled=compiled,
            )
    return info


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
    now = now or now_local()
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
