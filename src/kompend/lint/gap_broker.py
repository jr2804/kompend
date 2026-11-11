"""GapBroker — per-run state for the agentic remediation agent.

Holds the coverage-gap concepts that fuzzy-matched cached extraction
buckets, hands them out in batches via ``fetch()`` (same contract as
:class:`kompend.compilation.concept_broker.ConceptBroker`), and records
the agent's per-gap outcome: page generated (``mark_done``) or declined
with a reason (``mark_unfixable``).
"""

from __future__ import annotations

from collections.abc import Mapping

from kompend.agents.types import GapConcept
from kompend.types import JSONValue


class GapBroker:
    """Work queue over matched coverage-gap concept buckets."""

    def __init__(self, concepts: list[GapConcept]) -> None:
        self._concepts = list(concepts)
        self._cursor = 0
        self.done: list[str] = []
        self.unfixable: dict[str, str] = {}

    def remaining(self) -> int:
        """Number of not-yet-fetched gap concepts."""
        return max(len(self._concepts) - self._cursor, 0)

    def fetch(self, caller_id: str, limit: int = 5) -> list[GapConcept]:
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
        """Record that *name* could not be remediated, and why."""
        self.unfixable[name] = reason


def gap_concept(page: Mapping[str, JSONValue]) -> GapConcept:
    """Narrow one ``concept_pages()`` record to :class:`GapConcept`.

    The extraction layer hands back plain dicts; this is the one place that
    decides what a bucket is, so the remediation pass can read it without
    repeating the narrowing at every use.
    """
    name = page.get("name")
    return GapConcept(
        name=name if isinstance(name, str) else "",
        sources=_text_list(page.get("sources")),
        definitions=_text_list(page.get("definitions")),
        summaries=_text_list(page.get("summaries")),
    )


def _text_list(value: JSONValue | None) -> list[str]:
    """One bucket field read as a list of text, empty when it holds anything else."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
