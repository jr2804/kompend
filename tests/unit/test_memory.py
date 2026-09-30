"""Tests for KNOX Query Memory layer."""

from pathlib import Path

import pytest

from knox.memory.lifecycle import MemoryStore as LifecycleStore
from knox.memory.lifecycle import NodeState
from knox.memory.store import MemoryEntry, MemoryField, MemoryStore
from tests.conftest import DeterministicTestEmbeddingProvider

_TEST_PROVIDER = DeterministicTestEmbeddingProvider(dim=384)


class TestMemoryStore:
    """Tests for MemoryStore."""

    def test_add_entry_writes_log_and_zvec(self, temp_wiki_dir: Path) -> None:
        """Adding an entry should write to both the memory_log table and zvec."""
        store = MemoryStore(base_dir=temp_wiki_dir, embedding_provider=_TEST_PROVIDER)
        try:
            store.add_entry(query="What is the capital of France?", answer="Paris", feedback="Correct and concise.")

            entries = _log(temp_wiki_dir)
            assert len(entries) == 1
            data = entries[0]
            assert data[MemoryField.QUERY.value] == "What is the capital of France?"
            assert data[MemoryField.ANSWER.value] == "Paris"
            assert data[MemoryField.FEEDBACK.value] == "Correct and concise."
            assert MemoryField.TIMESTAMP in data

            # Check zvec search works
            results = store.search(query="capital france", top_k=1)
            assert len(results) == 1
            assert results[0][MemoryField.QUERY.value] == "What is the capital of France?"
            assert results[0][MemoryField.ANSWER.value] == "Paris"
        finally:
            store.close()

    def test_search_returns_top_k(self, temp_wiki_dir: Path) -> None:
        """Search should return the specified number of top results."""
        store = MemoryStore(base_dir=temp_wiki_dir, embedding_provider=_TEST_PROVIDER)
        try:
            store.add_entry(query="apple fruit", answer="A red fruit")
            store.add_entry(query="apple company", answer="A tech company")
            store.add_entry(query="banana fruit", answer="A yellow fruit")

            results = store.search(query="apple", top_k=2)
            assert len(results) == 2
            # Both apple entries should be in the top 2
            queries = [r[MemoryField.QUERY.value] for r in results]
            assert "apple fruit" in queries
            assert "apple company" in queries
        finally:
            store.close()

    def test_get_recent_entries_returns_chronological(self, temp_wiki_dir: Path) -> None:
        """get_recent_entries should return entries in reverse chronological order."""
        store = MemoryStore(base_dir=temp_wiki_dir, embedding_provider=_TEST_PROVIDER)
        try:
            store.add_entry(query="first", answer="1")
            store.add_entry(query="second", answer="2")
            store.add_entry(query="third", answer="3")

            recent = store.get_recent_entries(limit=2)
            assert len(recent) == 2
            # Should be newest first
            assert recent[0][MemoryField.QUERY.value] == "third"
            assert recent[1][MemoryField.QUERY.value] == "second"
        finally:
            store.close()

    def test_empty_store_handles_gracefully(self, temp_wiki_dir: Path) -> None:
        """An empty store should return empty results without errors."""
        store = MemoryStore(base_dir=temp_wiki_dir, embedding_provider=_TEST_PROVIDER)
        try:
            assert store.search(query="nothing", top_k=5) == []
            assert store.get_recent_entries(limit=10) == []
        finally:
            store.close()


class TestUserTracking:
    """memory_by provenance and store reopen (knox-gsb5)."""

    def test_add_entry_records_memory_by(self, temp_wiki_dir: Path) -> None:
        store = MemoryStore(base_dir=temp_wiki_dir, embedding_provider=_TEST_PROVIDER)
        try:
            store.add_entry(query="q", answer="a", memory_by="alice")
        finally:
            store.close()
        assert _log(temp_wiki_dir)[0][MemoryField.MEMORY_BY.value] == "alice"

    def test_add_entry_default_memory_by_empty(self, temp_wiki_dir: Path) -> None:
        store = MemoryStore(base_dir=temp_wiki_dir, embedding_provider=_TEST_PROVIDER)
        try:
            store.add_entry(query="q", answer="a")
        finally:
            store.close()
        assert _log(temp_wiki_dir)[0][MemoryField.MEMORY_BY.value] == ""

    def test_store_reopens_across_instances(self, temp_wiki_dir: Path) -> None:
        """Regression: a second MemoryStore on the same path must not crash.

        Previously ``create_and_open`` raised on an existing path, so every
        ``knox memory add`` after the first failed before writing the log.
        """
        s1 = MemoryStore(base_dir=temp_wiki_dir, embedding_provider=_TEST_PROVIDER)
        s1.add_entry(query="q1", answer="a1", memory_by="alice")
        s1.close()

        s2 = MemoryStore(base_dir=temp_wiki_dir, embedding_provider=_TEST_PROVIDER)
        try:
            s2.add_entry(query="q2", answer="a2", memory_by="bob")
        finally:
            s2.close()

        log_entries = _log(temp_wiki_dir)
        assert [e[MemoryField.QUERY.value] for e in log_entries] == ["q1", "q2"]
        assert [e[MemoryField.MEMORY_BY.value] for e in log_entries] == ["alice", "bob"]

    def test_get_recent_entries_includes_memory_by(self, temp_wiki_dir: Path) -> None:
        store = MemoryStore(base_dir=temp_wiki_dir, embedding_provider=_TEST_PROVIDER)
        try:
            store.add_entry(query="q", answer="a", memory_by="carol")
            recent = store.get_recent_entries(limit=5)
            assert recent[0][MemoryField.MEMORY_BY.value] == "carol"
        finally:
            store.close()


class TestLifecycleStore:
    """Memory lifecycle state persists to the vault store."""

    def test_save_load_roundtrip(self, temp_wiki_dir: Path) -> None:
        store_file = temp_wiki_dir / ".knox" / "memory.json"
        original = LifecycleStore(
            states={
                "n1": NodeState(node_id="n1", title="Alpha", confidence=0.8, tier="episodic", review_count=3),
                "n2": NodeState(node_id="n2", title="Beta", superseded_by="n1"),
            }
        )
        original.save(store_file)

        loaded = LifecycleStore.load(store_file)
        assert set(loaded.states) == {"n1", "n2"}
        n1 = loaded.get("n1")
        n2 = loaded.get("n2")
        assert n1 is not None
        assert n2 is not None
        assert n1.confidence == 0.8
        assert n1.tier == "episodic"
        assert n1.review_count == 3
        assert n2.is_superseded()

    def test_load_absent_store_is_empty(self, temp_wiki_dir: Path) -> None:
        loaded = LifecycleStore.load(temp_wiki_dir / ".knox" / "memory.json")
        assert loaded.states == {}


def _log(temp_wiki_dir: Path, limit: int = 20) -> list[MemoryEntry]:
    """Chronological log entries, oldest first, from the vault store."""
    return list(reversed(MemoryStore(base_dir=temp_wiki_dir).get_recent_entries(limit=limit)))


@pytest.fixture
def temp_wiki_dir(tmp_path: Path) -> Path:
    """Provide a temporary wiki directory structure."""
    wiki_dir = tmp_path / "test_wiki"
    wiki_dir.mkdir()
    (wiki_dir / "sources").mkdir()
    (wiki_dir / "wiki").mkdir()
    return wiki_dir
