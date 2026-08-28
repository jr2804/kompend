"""GapBroker — per-run state for the agentic remediation agent.

Holds the coverage-gap concepts that fuzzy-matched cached extraction
buckets, hands them out in batches via ``fetch()`` (same contract as
:class:`knox.compilation.concept_broker.ConceptBroker`), and records
the agent's per-gap outcome: page generated (``mark_done``) or declined
with a reason (``mark_unfixable``).
"""

from __future__ import annotations

from typing import Any


class GapBroker:
    """Work queue over matched coverage-gap concept buckets."""

    def __init__(self, concepts: list[dict[str, Any]]) -> None:
        self._concepts = list(concepts)
        self._cursor = 0
        self.done: list[str] = []
        self.unfixable: dict[str, str] = {}

    def remaining(self) -> int:
        """Number of not-yet-fetched gap concepts."""
        return max(len(self._concepts) - self._cursor, 0)

    def fetch(self, caller_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return the next *limit* gap-concept buckets.

        Returns an empty list when all gaps have been handed out.
        ``caller_id`` is accepted for broker-API parity but ignored —
        remediation runs a single agent over the whole queue.
        """
        _ = caller_id
        if self._cursor >= len(self._concepts):
            return []
        batch = self._concepts[self._cursor : self._cursor + max(limit, 1)]
        self._cursor += len(batch)
        return batch

    def mark_done(self, name: str) -> None:
        """Record that a page was generated for *name*."""
        if name not in self.done:
            self.done.append(name)

    def mark_unfixable(self, name: str, reason: str) -> None:
        """Record that *name* was declined, with the agent's reason."""
        self.unfixable[name] = reason
