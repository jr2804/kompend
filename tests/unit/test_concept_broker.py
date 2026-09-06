"""Unit tests for ConceptBroker concept assignment and diagnostics."""

from __future__ import annotations

from knox.compilation.concept_broker import ConceptBroker


def test_remaining_concepts_tracks_unassigned() -> None:
    broker = ConceptBroker(_concepts(10), num_callers=2)
    seg0 = ConceptBroker.caller_id(0)
    seg1 = ConceptBroker.caller_id(1)

    # Contiguous split: seg0 gets concepts 0-4, seg1 gets 5-9.
    assert broker.remaining(seg0) == 5
    assert broker.remaining(seg1) == 5

    broker.fetch(seg0, limit=2)
    remaining = broker.remaining_concepts(seg0)
    assert len(remaining) == 3
    assert [c["name"] for c in remaining] == ["concept-2", "concept-3", "concept-4"]

    # seg1 untouched.
    assert len(broker.remaining_concepts(seg1)) == 5


def test_remaining_concepts_empty_when_drained() -> None:
    broker = ConceptBroker(_concepts(3), num_callers=1)
    seg0 = ConceptBroker.caller_id(0)
    broker.fetch(seg0, limit=10)
    assert broker.remaining(seg0) == 0
    assert broker.remaining_concepts(seg0) == []


def test_remaining_concepts_unknown_caller() -> None:
    broker = ConceptBroker(_concepts(3), num_callers=1)
    assert broker.remaining_concepts("nope") == []


def test_requeue_puts_uncovered_back_in_pool() -> None:
    """Requeued concepts are handed out again on the next fetch."""
    broker = ConceptBroker(_concepts(4), num_callers=1)
    seg0 = ConceptBroker.caller_id(0)

    first = broker.fetch(seg0, limit=2)
    assert [c["name"] for c in first] == ["concept-0", "concept-1"]

    # Model produced pages only for concept-0 — requeue concept-1.
    broker.requeue(seg0, [first[1]])
    assert [c["name"] for c in broker.consumed(seg0)] == ["concept-0"]
    assert [c["name"] for c in broker.remaining_concepts(seg0)] == ["concept-1", "concept-2", "concept-3"]

    # Next fetch hands the requeued concept out first.
    second = broker.fetch(seg0, limit=1)
    assert [c["name"] for c in second] == ["concept-1"]


def test_requeue_empty_is_noop() -> None:
    broker = ConceptBroker(_concepts(2), num_callers=1)
    seg0 = ConceptBroker.caller_id(0)
    broker.requeue(seg0, [])
    assert broker.remaining(seg0) == 2


def _concepts(n: int) -> list[dict[str, str]]:
    return [{"name": f"concept-{i}"} for i in range(n)]
