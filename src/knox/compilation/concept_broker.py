"""ConceptBroker — per-run state that hands disjoint concept batches to segment agents.

Each generation run creates one ``ConceptBroker`` shared across all per-segment
agents.  On init it round-robins the concept list across segment ``caller_id``
strings so every agent gets a fair, disjoint subset with no coordination.

Thread safety is *not* required: agents run sequentially.
"""

from __future__ import annotations

from typing import Any


class ConceptBroker:
    """Holds all concepts and tracks which caller has been assigned what.

    ``fetch(caller_id, limit)`` returns the next *limit* unassigned concepts
    for that caller and marks them assigned.  Returns ``[]`` when the caller
    has exhausted its pre-allocated list.
    """

    def __init__(self, concepts: list[dict[str, Any]], num_callers: int = 1) -> None:
        self._concepts = concepts
        total = len(concepts)
        num_callers = max(num_callers, 1)

        # Pre-allocate: round-robin caller -> list of concept dicts
        self._per_caller: dict[str, list[dict[str, Any]]] = {}
        self._cursors: dict[str, int] = {}
        for i in range(num_callers):
            key = _caller_id(i)
            # Fair share — earlier callers get the surplus.
            start = i * (total // num_callers) + min(i, total % num_callers)
            end = start + (total // num_callers) + (1 if i < total % num_callers else 0)
            self._per_caller[key] = list(concepts[start:end])
            self._cursors[key] = 0

    def fetch(self, caller_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return the next *limit* concepts for *caller_id*.

        Returns an empty list when all concepts for this caller have
        been exhausted.
        """
        pool = self._per_caller.get(caller_id)
        if pool is None:
            return []
        cursor = self._cursors.get(caller_id, 0)
        if cursor >= len(pool):
            return []
        batch = pool[cursor : cursor + limit]
        self._cursors[caller_id] = cursor + len(batch)
        return batch

    def remaining(self, caller_id: str) -> int:
        """Number of unassigned concepts remaining for *caller_id*."""
        pool = self._per_caller.get(caller_id, [])
        cursor = self._cursors.get(caller_id, 0)
        return max(0, len(pool) - cursor)

    def remaining_concepts(self, caller_id: str) -> list[dict[str, Any]]:
        """Unassigned concept dicts remaining for *caller_id* (for diagnostics)."""
        pool = self._per_caller.get(caller_id, [])
        cursor = self._cursors.get(caller_id, 0)
        return list(pool[cursor:])

    def consumed(self, caller_id: str) -> list[dict[str, Any]]:
        """Concept dicts already handed to *caller_id* (fetched, assigned)."""
        pool = self._per_caller.get(caller_id, [])
        cursor = self._cursors.get(caller_id, 0)
        return list(pool[:cursor])

    def mark_caller_consumed(self, caller_id: str, names: list[str]) -> None:
        """Advance *caller_id*'s cursor past the concepts whose *names* are given.

        Used at checkpoint resume to skip already-produced pages: the
        broker's pre-allocation of concepts to segments is deterministic
        (segment index → contiguous slice), so advancing the cursor past
        the already-finished names is sufficient — the next ``fetch`` call
        hands out only the remaining unconsumed concepts for that segment.

        Concepts not found in the caller's pool are ignored silently
        (covers checkpoint mismatches across dep versions).
        """
        if not names:
            return
        pool = self._per_caller.get(caller_id)
        if pool is None:
            return
        target = {str(n) for n in names}
        cursor = self._cursors.get(caller_id, 0)
        while cursor < len(pool) and str(pool[cursor].get("name", "")) in target:
            cursor += 1
        # Continue past any already-skipped names so callers using a
        # smaller name set still leave the cursor at the first
        # unconsumed concept.
        self._cursors[caller_id] = cursor

    def requeue(self, caller_id: str, concepts: list[dict[str, Any]]) -> None:
        """Put *concepts* back at the front of the caller's remaining pool.

        Used by the coverage loop when the agent fetched concepts but did
        not produce pages for all of them: requeued concepts are handed out
        again on the next pass, giving the model a second chance within the
        same bounded loop. Requeued concepts leave the consumed set.
        """
        if not concepts:
            return
        pool = self._per_caller.setdefault(caller_id, [])
        cursor = self._cursors.get(caller_id, 0)
        requeued_names = {c.get("name") for c in concepts}
        consumed = [c for c in pool[:cursor] if c.get("name") not in requeued_names]
        self._per_caller[caller_id] = consumed + concepts + pool[cursor:]
        self._cursors[caller_id] = len(consumed)

    @property
    def total_concepts(self) -> int:
        return len(self._concepts)

    @property
    def all_consumed(self) -> bool:
        return sum(self.remaining(k) for k in self._per_caller) == 0

    @classmethod
    def caller_id(cls, segment_index: int) -> str:
        """Canonical caller key for a segment index (single source of truth)."""
        return _caller_id(segment_index)


def _caller_id(segment_index: int) -> str:
    return f"seg_{segment_index}"
