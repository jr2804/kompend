"""Memory lifecycle: confidence, tiers, supersession, retention/decay.

Implements the Rohit-v2 memory lifecycle concepts:

* **Confidence scoring** — each wiki node has a confidence that evolves over
  successive compiles, asymptoting toward 1.0.
* **Consolidation tiers** — working → episodic → semantic → procedural,
  promoted automatically based on review count.
* **Supersession** — when a recompile produces a node with the same title but
  a different ID, the old node is marked as superseded.
* **Retention/decay** — exponential forgetting curve based on time since last
  review.  Nodes whose retention drops below a threshold become stale-page
  candidates.

State lives in the single vault store (``.kompend/kompend.db``, ``memory_states``
table); the legacy ``memory.json`` is no longer written or read.
"""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants;
# no runtime value ever enters an identifier or statement.

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from kompend.core.models import CompiledWiki, WikiNode
from kompend.core.paths import VaultPaths
from kompend.core.store import open_store, vault_db_path
from kompend.core.store.schema import MEMORY_STATES_TABLE
from kompend.core.utils.time import iso_local, now_local
from kompend.logging import get_logger

DEFAULT_HALF_LIFE_DAYS = 90.0
DEFAULT_DECAY_THRESHOLD = 0.3

_CONFIDENCE_BOOST = 0.15


class MemoryTier(StrEnum):
    """Memory consolidation tiers, ordered by increasing permanence."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


_TIER_PROMOTION_COUNT: dict[MemoryTier, int] = {
    MemoryTier.WORKING: 2,
    MemoryTier.EPISODIC: 5,
    MemoryTier.SEMANTIC: 10,
}

_TIER_ORDER: list[MemoryTier] = [
    MemoryTier.WORKING,
    MemoryTier.EPISODIC,
    MemoryTier.SEMANTIC,
    MemoryTier.PROCEDURAL,
]

logger = get_logger(__name__)


@dataclass
class NodeState:
    """Lifecycle state attached to a single wiki node."""

    node_id: str
    title: str
    confidence: float = 0.5
    tier: str = MemoryTier.WORKING.value
    birth_at: str = ""
    review_at: str = ""
    review_count: int = 0
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        now = _now_iso()
        if not self.birth_at:
            self.birth_at = now
        if not self.review_at:
            self.review_at = now

    def retention(
        self,
        *,
        now: datetime | None = None,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    ) -> float:
        """Exponential retention: ``confidence * exp(-elapsed / half_life)``."""
        now_dt = now or now_local()
        last_review = _parse_iso(self.review_at)
        elapsed_days = max(0.0, (now_dt - last_review).total_seconds() / 86400.0)
        return self.confidence * math.exp(-elapsed_days / half_life_days)

    def is_stale(
        self,
        *,
        threshold: float = DEFAULT_DECAY_THRESHOLD,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        now: datetime | None = None,
    ) -> bool:
        """Whether this node's retention has decayed below *threshold*."""
        return self.retention(now=now, half_life_days=half_life_days) < threshold

    def is_superseded(self) -> bool:
        return self.superseded_by is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeState:
        return cls(
            node_id=data["node_id"],
            title=data.get("title", ""),
            confidence=float(data.get("confidence", 0.5)),
            tier=data.get("tier", MemoryTier.WORKING.value),
            birth_at=data.get("birth_at", ""),
            review_at=data.get("review_at", ""),
            review_count=int(data.get("review_count", 0)),
            superseded_by=data.get("superseded_by"),
        )


@dataclass
class CompileReport:
    """Summary of what happened during a memory update."""

    new: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)


@dataclass
class MemoryStore:
    """Persistent collection of :class:`NodeState` records."""

    states: dict[str, NodeState] = field(default_factory=dict)

    def save(self, store_file: Path) -> None:
        """Persist the memory store to the vault store.

        *store_file* may be the vault ``base_dir`` or the legacy
        ``.kompend/memory.json`` path; both resolve to the same ``kompend.db``. The
        ``memory_states`` table is replaced with the in-memory set.
        """
        db_path = vault_db_path(store_file)
        rows = [(s.node_id, s.title, s.confidence, s.tier, s.birth_at, s.review_at, s.review_count, s.superseded_by) for s in self.states.values()]
        with open_store(db_path) as conn:
            conn.execute(f"DELETE FROM {MEMORY_STATES_TABLE}")
            conn.executemany(
                f"INSERT INTO {MEMORY_STATES_TABLE}(node_id, title, confidence, tier, birth_at, review_at, review_count, superseded_by) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    # ── Access ────────────────────────────────────────────────────────

    def get(self, node_id: str) -> NodeState | None:
        return self.states.get(node_id)

    def all_stale(
        self,
        *,
        threshold: float = DEFAULT_DECAY_THRESHOLD,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        now: datetime | None = None,
    ) -> list[NodeState]:
        """Return states whose retention has decayed below *threshold*."""
        return [s for s in self.states.values() if not s.is_superseded() and s.is_stale(threshold=threshold, half_life_days=half_life_days, now=now)]

    def all_superseded(self) -> list[NodeState]:
        """Return all superseded states."""
        return [s for s in self.states.values() if s.is_superseded()]

    # ── Update ────────────────────────────────────────────────────────

    def update_for_compile(
        self,
        wiki_nodes: list[WikiNode],
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        decay_threshold: float = DEFAULT_DECAY_THRESHOLD,
    ) -> CompileReport:
        """Update the store after a fresh compile.

        Detects supersession (same title, different node ID), promotes tiers,
        and evolves confidence for re-seen nodes.

        Parameters
        ----------
        wiki_nodes:
            List of objects with ``.id``, ``.title``, and ``.metadata``
            attributes (i.e. :class:`~kompend.core.models.WikiNode`).
        half_life_days:
            Retention half-life in days (used only for the decay report).
        decay_threshold:
            Retention threshold below which a node is considered stale.

        Returns:
        -------
        CompileReport
            Lists of new, updated, and superseded node IDs.
        """
        report = CompileReport()
        now = _now_iso()

        title_to_old: dict[str, NodeState] = {}
        for old_state in self.states.values():
            if not old_state.is_superseded():
                title_to_old.setdefault(old_state.title, old_state)

        seen_ids: set[str] = set()

        for node in wiki_nodes:
            node_id = node.id
            title = node.title
            seen_ids.add(node_id)

            existing = self.states.get(node_id)

            if existing is not None and not existing.is_superseded():
                # Node re-compiled with same ID — bump confidence and review.
                new_confidence = existing.confidence + _CONFIDENCE_BOOST * (1.0 - existing.confidence)
                new_confidence = min(1.0, new_confidence)
                new_review_count = existing.review_count + 1
                new_tier = promote_tier(existing.tier, new_review_count)
                self.states[node_id] = NodeState(
                    node_id=node_id,
                    title=title,
                    confidence=new_confidence,
                    tier=new_tier,
                    birth_at=existing.birth_at,
                    review_at=now,
                    review_count=new_review_count,
                )
                report.updated.append(node_id)
            else:
                # New node. Check if a same-title node existed before.
                old_by_title = title_to_old.get(title)
                if old_by_title is not None and old_by_title.node_id != node_id:
                    # Supersession: old node replaced by new.
                    old_by_title.superseded_by = node_id
                    report.superseded.append(old_by_title.node_id)
                    logger.info("Node %r superseded by %r", old_by_title.node_id, node_id)

                self.states[node_id] = NodeState(
                    node_id=node_id,
                    title=title,
                    confidence=0.5,
                )
                report.new.append(node_id)

        return report

    # ── Persistence ───────────────────────────────────────────────────

    @classmethod
    def load(cls, store_file: Path) -> MemoryStore:
        """Load a memory store from the vault store.

        *store_file* may be the vault ``base_dir`` or the legacy
        ``.kompend/memory.json`` path; both resolve to the same ``kompend.db``.
        Returns an empty store when the database does not exist yet.
        """
        db_path = vault_db_path(store_file)
        if not db_path.exists():
            return cls()
        with open_store(db_path) as conn:
            rows = conn.execute(f"SELECT * FROM {MEMORY_STATES_TABLE}").fetchall()
        states: dict[str, NodeState] = {}
        for row in rows:
            state = NodeState(
                node_id=row["node_id"],
                title=row["title"],
                confidence=float(row["confidence"]),
                tier=row["tier"],
                birth_at=row["birth_at"],
                review_at=row["review_at"],
                review_count=int(row["review_count"]),
                superseded_by=row["superseded_by"],
            )
            states[state.node_id] = state
        return cls(states=states)


def _now_iso() -> str:
    return iso_local(timespec="microseconds")


def _parse_iso(ts: str) -> datetime:
    if not ts:
        return now_local()
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return now_local()


def promote_tier(current: str, review_count: int) -> MemoryTier:
    """Return the tier a node should have given its review count."""
    tier: MemoryTier = MemoryTier(current)
    while tier in _TIER_PROMOTION_COUNT and review_count >= _TIER_PROMOTION_COUNT[tier]:
        idx = _TIER_ORDER.index(tier)
        if idx + 1 < len(_TIER_ORDER):
            tier = _TIER_ORDER[idx + 1]
        else:
            break
    return tier


def update_memory_for_compile(
    wiki: CompiledWiki,
    base_dir: Path,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    decay_threshold: float = DEFAULT_DECAY_THRESHOLD,
) -> CompileReport:
    """Load, update, and persist the memory store for *wiki*.

    The store is saved alongside other kompend metadata (``.kompend/`` directory).
    """
    store_file = VaultPaths(base_dir).memory_store_file
    store = MemoryStore.load(store_file)
    report = store.update_for_compile(
        wiki.nodes,
        half_life_days=half_life_days,
        decay_threshold=decay_threshold,
    )
    store.save(store_file)

    if report.new or report.updated or report.superseded:
        logger.info(
            "Memory update: %d new, %d updated, %d superseded",
            len(report.new),
            len(report.updated),
            len(report.superseded),
        )

    # Attach memory metadata to wiki for downstream consumers.
    wiki.metadata.setdefault("memory", {})
    wiki.metadata["memory"]["new"] = report.new
    wiki.metadata["memory"]["updated"] = report.updated
    wiki.metadata["memory"]["superseded"] = report.superseded
    wiki.metadata["memory"]["stale"] = [s.node_id for s in store.all_stale(threshold=decay_threshold, half_life_days=half_life_days)]

    return report
