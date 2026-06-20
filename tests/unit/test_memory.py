"""Tests for KNOX Query Memory layer."""

import json
from pathlib import Path

import pytest

from knox.core.paths import KNOX_DIR, MEMORY_LOG_FILE
from knox.memory.store import MemoryStore
from tests.conftest import DeterministicTestEmbeddingProvider

_TEST_PROVIDER = DeterministicTestEmbeddingProvider(dim=384)


@pytest.fixture
def temp_wiki_dir(tmp_path: Path) -> Path:
    """Provide a temporary wiki directory structure."""
    wiki_dir = tmp_path / "test_wiki"
    wiki_dir.mkdir()
    (wiki_dir / "sources").mkdir()
    (wiki_dir / "wiki").mkdir()
    return wiki_dir


class TestMemoryStore:
    """Tests for MemoryStore."""

    def test_add_entry_creates_jsonl_and_zvec(self, temp_wiki_dir: Path) -> None:
        """Adding an entry should write to both JSONL and zvec."""
        store = MemoryStore(base_dir=temp_wiki_dir, embedding_provider=_TEST_PROVIDER)
        try:
            store.add_entry(query="What is the capital of France?", answer="Paris", feedback="Correct and concise.")

            # Check JSONL
            log_path = temp_wiki_dir / KNOX_DIR / MEMORY_LOG_FILE
            assert log_path.exists()
            with open(log_path, encoding="utf-8") as f:
                line = f.readline()
                data = json.loads(line)
                assert data["query"] == "What is the capital of France?"
                assert data["answer"] == "Paris"
                assert data["feedback"] == "Correct and concise."
                assert "timestamp" in data

            # Check zvec search works
            results = store.search(query="capital france", top_k=1)
            assert len(results) == 1
            assert results[0]["query"] == "What is the capital of France?"
            assert results[0]["answer"] == "Paris"
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
            queries = [r["query"] for r in results]
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
            assert recent[0]["query"] == "third"
            assert recent[1]["query"] == "second"
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
        log_path = temp_wiki_dir / KNOX_DIR / MEMORY_LOG_FILE
        data = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
        assert data["memory_by"] == "alice"

    def test_add_entry_default_memory_by_empty(self, temp_wiki_dir: Path) -> None:
        store = MemoryStore(base_dir=temp_wiki_dir, embedding_provider=_TEST_PROVIDER)
        try:
            store.add_entry(query="q", answer="a")
        finally:
            store.close()
        log_path = temp_wiki_dir / KNOX_DIR / MEMORY_LOG_FILE
        data = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
        assert data["memory_by"] == ""

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

        log_path = temp_wiki_dir / KNOX_DIR / MEMORY_LOG_FILE
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert [e["query"] for e in entries] == ["q1", "q2"]
        assert [e["memory_by"] for e in entries] == ["alice", "bob"]

    def test_get_recent_entries_includes_memory_by(self, temp_wiki_dir: Path) -> None:
        store = MemoryStore(base_dir=temp_wiki_dir, embedding_provider=_TEST_PROVIDER)
        try:
            store.add_entry(query="q", answer="a", memory_by="carol")
            recent = store.get_recent_entries(limit=5)
            assert recent[0]["memory_by"] == "carol"
        finally:
            store.close()
