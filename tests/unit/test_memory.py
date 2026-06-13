"""Tests for the memory lifecycle module (knox-rqt)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from knox.core.memory import (
    MemoryStore,
    MemoryTier,
    NodeState,
    promote_tier,
    update_memory_for_compile,
)


@dataclass
class FakeNode:
    id: str
    title: str
    content: str = ""
    node_type: str = "concept"
    source_chunks: list[str] | None = None
    metadata: dict[str, Any] = None


class TestNodeState:
    def test_initial_values(self) -> None:
        state = NodeState(node_id="n1", title="Test")
        assert state.confidence == 0.5
        assert state.tier == MemoryTier.WORKING.value
        assert state.review_count == 0
        assert state.superseded_by is None
        assert state.birth_at != ""
        assert state.review_at != ""

    def test_retention_at_birth(self) -> None:
        """A fresh node should have retention equal to its confidence."""
        now = datetime.now(UTC)
        state = NodeState(node_id="n1", title="Test", confidence=0.8, review_at=now.isoformat())
        ret = state.retention(now=now, half_life_days=90.0)
        assert ret == pytest.approx(0.8, abs=0.01)

    def test_retention_decay_over_time(self) -> None:
        """Retention should decrease as time passes."""
        now = datetime.now(UTC)
        review = now - timedelta(days=90)
        state = NodeState(node_id="n1", title="Test", confidence=1.0, review_at=review.isoformat())
        ret = state.retention(now=now, half_life_days=90.0)
        assert ret == pytest.approx(0.3679, abs=0.05)

    def test_is_stale_when_decayed(self) -> None:
        now = datetime.now(UTC)
        review = now - timedelta(days=365)
        state = NodeState(node_id="n1", title="Test", confidence=0.5, review_at=review.isoformat())
        assert state.is_stale(threshold=0.3, half_life_days=90.0, now=now)

    def test_not_stale_when_recent(self) -> None:
        now = datetime.now(UTC)
        state = NodeState(node_id="n1", title="Test", confidence=0.9, review_at=now.isoformat())
        assert not state.is_stale(threshold=0.3, half_life_days=90.0, now=now)

    def test_superseded_flag(self) -> None:
        state = NodeState(node_id="n1", title="Old", superseded_by="n2")
        assert state.is_superseded()
        state2 = NodeState(node_id="n2", title="New")
        assert not state2.is_superseded()

    def test_serialization_roundtrip(self) -> None:
        state = NodeState(node_id="n1", title="Test", confidence=0.7, tier=MemoryTier.SEMANTIC.value, review_count=5)
        d = state.to_dict()
        restored = NodeState.from_dict(d)
        assert restored.node_id == "n1"
        assert restored.confidence == 0.7
        assert restored.tier == MemoryTier.SEMANTIC.value
        assert restored.review_count == 5


class TestPromoteTier:
    def test_working_to_episodic(self) -> None:
        assert promote_tier(MemoryTier.WORKING.value, 2) == MemoryTier.EPISODIC.value

    def test_episodic_to_semantic(self) -> None:
        assert promote_tier(MemoryTier.EPISODIC.value, 5) == MemoryTier.SEMANTIC.value

    def test_semantic_to_procedural(self) -> None:
        assert promote_tier(MemoryTier.SEMANTIC.value, 10) == MemoryTier.PROCEDURAL.value

    def test_procedural_stays(self) -> None:
        assert promote_tier(MemoryTier.PROCEDURAL.value, 100) == MemoryTier.PROCEDURAL.value

    def test_no_promotion_below_threshold(self) -> None:
        assert promote_tier(MemoryTier.WORKING.value, 1) == MemoryTier.WORKING.value


class TestMemoryStore:
    def test_load_missing_file(self, tmp_path: Path) -> None:
        store = MemoryStore.load(tmp_path / "nonexistent.json")
        assert len(store.states) == 0

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        store = MemoryStore()
        store.states["n1"] = NodeState(node_id="n1", title="Alpha", confidence=0.9)
        path = tmp_path / "memory.json"
        store.save(path)
        assert path.exists()

        loaded = MemoryStore.load(path)
        assert "n1" in loaded.states
        assert loaded.states["n1"].title == "Alpha"
        assert loaded.states["n1"].confidence == 0.9

    def test_load_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.json"
        path.write_text("{invalid json", encoding="utf-8")
        store = MemoryStore.load(path)
        assert len(store.states) == 0

    def test_first_compile_creates_new_nodes(self) -> None:
        store = MemoryStore()
        nodes = [FakeNode(id="n1", title="A"), FakeNode(id="n2", title="B")]
        report = store.update_for_compile(nodes)
        assert report.new == ["n1", "n2"]
        assert report.updated == []
        assert report.superseded == []
        assert store.states["n1"].confidence == 0.5
        assert store.states["n1"].tier == MemoryTier.WORKING.value

    def test_recompile_increases_confidence(self) -> None:
        store = MemoryStore()
        nodes = [FakeNode(id="n1", title="A")]
        store.update_for_compile(nodes)
        first_conf = store.states["n1"].confidence

        store.update_for_compile(nodes)
        second_conf = store.states["n1"].confidence

        assert second_conf > first_conf
        assert store.states["n1"].review_count == 1

    def test_confidence_asymptotes_to_one(self) -> None:
        store = MemoryStore()
        nodes = [FakeNode(id="n1", title="A")]
        for _ in range(50):
            store.update_for_compile(nodes)
        assert store.states["n1"].confidence > 0.99

    def test_tier_promotion_on_recompile(self) -> None:
        store = MemoryStore()
        nodes = [FakeNode(id="n1", title="A")]
        store.update_for_compile(nodes)
        assert store.states["n1"].tier == MemoryTier.WORKING.value

        store.update_for_compile(nodes)
        store.update_for_compile(nodes)
        assert store.states["n1"].tier == MemoryTier.EPISODIC.value

    def test_supersession_detection(self) -> None:
        store = MemoryStore()
        old_nodes = [FakeNode(id="old-1", title="Transformer")]
        store.update_for_compile(old_nodes)

        new_nodes = [FakeNode(id="new-1", title="Transformer")]
        report = store.update_for_compile(new_nodes)

        assert report.superseded == ["old-1"]
        assert store.states["old-1"].superseded_by == "new-1"
        assert "new-1" in store.states

    def test_no_supersession_same_id(self) -> None:
        store = MemoryStore()
        nodes = [FakeNode(id="n1", title="Alpha")]
        store.update_for_compile(nodes)
        report = store.update_for_compile(nodes)
        assert report.superseded == []

    def test_all_stale(self) -> None:
        now = datetime.now(UTC)
        store = MemoryStore()
        store.states["stale"] = NodeState(
            node_id="stale",
            title="Old",
            confidence=0.1,
            review_at=(now - timedelta(days=365)).isoformat(),
        )
        store.states["fresh"] = NodeState(
            node_id="fresh",
            title="New",
            confidence=0.9,
            review_at=now.isoformat(),
        )
        stale = store.all_stale(threshold=0.3, half_life_days=90.0, now=now)
        assert len(stale) == 1
        assert stale[0].node_id == "stale"

    def test_all_superseded_excluded_from_stale(self) -> None:
        now = datetime.now(UTC)
        store = MemoryStore()
        store.states["old"] = NodeState(
            node_id="old",
            title="Old",
            confidence=0.1,
            review_at=(now - timedelta(days=365)).isoformat(),
            superseded_by="new",
        )
        stale = store.all_stale(threshold=0.3, half_life_days=90.0, now=now)
        assert len(stale) == 0

    def test_all_superseded_list(self) -> None:
        store = MemoryStore()
        store.states["a"] = NodeState(node_id="a", title="A", superseded_by="b")
        store.states["b"] = NodeState(node_id="b", title="B")
        sup = store.all_superseded()
        assert len(sup) == 1
        assert sup[0].node_id == "a"


class TestUpdateMemoryForCompile:
    def test_creates_memory_file(self, tmp_path: Path) -> None:
        @dataclass
        class FakeWiki:
            nodes: list[Any]
            edges: list[Any]
            toc: list[Any]
            metadata: dict[str, Any]

        wiki = FakeWiki(
            nodes=[FakeNode(id="n1", title="Alpha"), FakeNode(id="n2", title="Beta")],
            edges=[],
            toc=[],
            metadata={},
        )
        report = update_memory_for_compile(wiki, tmp_path)

        assert set(report.new) == {"n1", "n2"}
        assert (tmp_path / ".knox" / "memory.json").exists()
        assert "memory" in wiki.metadata
        assert "new" in wiki.metadata["memory"]

    def test_supersession_propagated_to_metadata(self, tmp_path: Path) -> None:
        @dataclass
        class FakeWiki:
            nodes: list[Any]
            edges: list[Any]
            toc: list[Any]
            metadata: dict[str, Any]

        wiki1 = FakeWiki(nodes=[FakeNode(id="old-1", title="Gamma")], edges=[], toc=[], metadata={})
        update_memory_for_compile(wiki1, tmp_path)

        wiki2 = FakeWiki(nodes=[FakeNode(id="new-1", title="Gamma")], edges=[], toc=[], metadata={})
        update_memory_for_compile(wiki2, tmp_path)

        assert "old-1" in wiki2.metadata["memory"]["superseded"]
