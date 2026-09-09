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


def test_mark_caller_consumed_advances_cursor() -> None:
    """mark_caller_consumed skips ahead past the named concepts so fetch resumes from after them."""
    broker = ConceptBroker(_concepts(5), num_callers=1)
    seg0 = ConceptBroker.caller_id(0)

    broker.mark_caller_consumed(seg0, ["concept-0", "concept-1", "concept-2"])
    assert broker.remaining(seg0) == 2
    assert [c["name"] for c in broker.remaining_concepts(seg0)] == ["concept-3", "concept-4"]

    # fetch returns the next unconsumed concepts.
    nxt = broker.fetch(seg0, limit=10)
    assert [c["name"] for c in nxt] == ["concept-3", "concept-4"]
    assert broker.remaining(seg0) == 0


def test_mark_caller_consumed_unknown_names_are_skipped() -> None:
    """Names not in the caller's pool are ignored (covers checkpoint mismatches)."""
    broker = ConceptBroker(_concepts(3), num_callers=1)
    seg0 = ConceptBroker.caller_id(0)

    broker.mark_caller_consumed(seg0, ["nope", "concept-0"])
    assert broker.remaining(seg0) == 2
    assert [c["name"] for c in broker.remaining_concepts(seg0)] == ["concept-1", "concept-2"]


def test_mark_caller_consumed_empty_is_noop() -> None:
    broker = ConceptBroker(_concepts(2), num_callers=1)
    seg0 = ConceptBroker.caller_id(0)
    broker.mark_caller_consumed(seg0, [])
    assert broker.remaining(seg0) == 2


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


def test_format_uncovered_concepts_includes_sources() -> None:
    """Concepts are rendered with source provenance so warnings are traceable."""
    from knox.compilation.flows.generation import _format_uncovered_concepts

    concepts = [
        {"name": "PESQ score", "sources": ["ITU-T_P.862__2001.pdf"]},
        {"name": "VTR", "sources": ["ITU-T_P.863__03_2018.pdf", "ITU-T_P.Imp863.pdf"]},
        {"name": "G.191"},  # missing sources — must not silently drop (memory #149)
    ]
    broker = ConceptBroker(concepts, num_callers=1)
    seg0 = ConceptBroker.caller_id(0)
    out = _format_uncovered_concepts(broker, seg0, limit=10)
    assert "PESQ score [sources: ITU-T_P.862__2001.pdf]" in out
    assert "VTR [sources: ITU-T_P.862__2001.pdf" not in out  # not PESQ's source
    assert "ITU-T_P.863__03_2018.pdf" in out
    assert "G.191 [sources: (no-source)]" in out
    assert "(none)" not in out


def test_format_uncovered_concepts_truncates_with_ellipsis() -> None:
    from knox.compilation.flows.generation import _format_uncovered_concepts

    concepts = [{"name": f"c-{i}", "sources": []} for i in range(15)]
    broker = ConceptBroker(concepts, num_callers=1)
    seg0 = ConceptBroker.caller_id(0)
    out = _format_uncovered_concepts(broker, seg0, limit=3)
    assert out.count("[sources: (no-source)]") == 3
    assert out.endswith("…")


def test_format_uncovered_concepts_handles_string_sources() -> None:
    """A single source can be encoded as a bare string instead of a list."""
    from knox.compilation.flows.generation import _format_uncovered_concepts

    broker = ConceptBroker([{"name": "x", "sources": "single.pdf"}], num_callers=1)
    out = _format_uncovered_concepts(broker, ConceptBroker.caller_id(0))
    assert "x [sources: single.pdf]" in out

