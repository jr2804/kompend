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
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from knox.core.paths import VaultPaths
from knox.core.timeutil import iso_local, now_local
from knox.logging import get_logger

logger = get_logger(__name__)

DEFAULT_HALF_LIFE_DAYS = 90.0
DEFAULT_DECAY_THRESHOLD = 0.3

_CONFIDENCE_BOOST = 0.15

_TIER_PROMOTION_COUNT: dict[str, int] = {
    "working": 2,
    "episodic": 5,
    "semantic": 10,
}

_TIER_ORDER: list[str] = ["working", "episodic", "semantic", "procedural"]


class MemoryTier(StrEnum):
    """Memory consolidation tiers, ordered by increasing permanence."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


def _now_iso() -> str:
    return iso_local(timespec="microseconds")


def _parse_iso(ts: str) -> datetime:
    if not ts:
        return now_local()
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return now_local()


def promote_tier(current: str, review_count: int) -> str:
    """Return the tier a node should have given its review count."""
    tier = current
    while tier in _TIER_PROMOTION_COUNT and review_count >= _TIER_PROMOTION_COUNT[tier]:
        idx = _TIER_ORDER.index(tier)
        if idx + 1 < len(_TIER_ORDER):
            tier = _TIER_ORDER[idx + 1]
        else:
            break
    return tier


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
        """Persist the memory store to *store_file* as JSON."""
        store_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"states": [s.to_dict() for s in self.states.values()]}
        store_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

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
        wiki_nodes: list[Any],
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
            attributes (i.e. :class:`~knox.core.models.WikiNode`).
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
        """Load a memory store from a JSON file.

        Returns an empty store if the file does not exist or is corrupt.
        """
        if not store_file.exists():
            return cls()
        try:
            data = json.loads(store_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load memory store from %s: %s", store_file, exc, exc_info=True)
            return cls()
        raw_states = data.get("states", []) if isinstance(data, dict) else []
        states: dict[str, NodeState] = {}
        for raw in raw_states:
            if isinstance(raw, dict) and "node_id" in raw:
                state = NodeState.from_dict(raw)
                states[state.node_id] = state
        return cls(states=states)


def update_memory_for_compile(
    wiki: Any,
    output_dir: Path,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    decay_threshold: float = DEFAULT_DECAY_THRESHOLD,
) -> CompileReport:
    """Load, update, and persist the memory store for *wiki*.

    The store is saved alongside other KNOX metadata (``.knox/`` directory).
    """
    store_path = VaultPaths(output_dir.parent).memory_store_path
    store = MemoryStore.load(store_path)
    report = store.update_for_compile(
        wiki.nodes,
        half_life_days=half_life_days,
        decay_threshold=decay_threshold,
    )
    store.save(store_path)

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
