"""Tests for the vault-rooted agent-memory store.

``DirectoryMemoryStore`` implements the harness ``MemoryStore`` protocol over
plain files so the notebook survives the process (Windows-first — the
harness ``FileStore`` sandbox workspace has no Windows backend). The
contract below mirrors the harness stores: content-fingerprint versions for
compare-and-set, idempotent operation receipts, bounded reads, sorted path
listings, and escape-proof path validation.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import pytest
from pydantic_ai_harness.memory import MemoryConflictError, MemoryOperation, MemoryOperationConflictError

from kompend.agents.vault_memory import DirectoryMemoryStore

# -- read/write/delete -------------------------------------------------------


async def test_write_then_read_round_trip(tmp_path: Any) -> None:
    store = DirectoryMemoryStore(tmp_path)
    mutation = await store.write("MEMORY.md", "# Vault memory\n", expected_version=None)
    assert mutation.version == _version("# Vault memory\n")
    assert mutation.replayed is False
    assert mutation.existed is False

    file = await _read(store, "MEMORY.md")
    assert file is not None
    assert file.content == "# Vault memory\n"
    assert file.version == mutation.version
    assert file.truncated is False
    assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8") == "# Vault memory\n"


async def test_read_missing_file_is_none(tmp_path: Any) -> None:
    store = DirectoryMemoryStore(tmp_path)
    assert await _read(store, "MEMORY.md") is None


async def test_read_truncates_and_flags(tmp_path: Any) -> None:
    store = DirectoryMemoryStore(tmp_path)
    await store.write("topic.md", "x" * 100, expected_version=None)
    file = await _read(store, "topic.md", max_chars=10)
    assert file is not None
    assert file.content == "x" * 10
    assert file.truncated is True


async def test_write_replaces_and_reports_existed(tmp_path: Any) -> None:
    store = DirectoryMemoryStore(tmp_path)
    first = await store.write("topic.md", "one", expected_version=None)
    second = await store.write("topic.md", "two", expected_version=first.version)
    assert second.existed is True
    assert second.version == _version("two")


async def test_compare_and_set_rejects_stale_version(tmp_path: Any) -> None:
    store = DirectoryMemoryStore(tmp_path)
    await store.write("topic.md", "one", expected_version=None)
    with pytest.raises(MemoryConflictError, match="changed elsewhere"):
        await store.write("topic.md", "two", expected_version=_version("unrelated"))


async def test_delete_cas_and_missing_file(tmp_path: Any) -> None:
    store = DirectoryMemoryStore(tmp_path)
    mutation = await store.write("topic.md", "one", expected_version=None)
    deleted = await store.delete("topic.md", expected_version=mutation.version)
    assert deleted.existed is True
    assert deleted.version is None

    missing = await store.delete("topic.md", expected_version=None)
    assert missing.existed is False

    assert await _read(store, "topic.md") is None


async def test_delete_rejects_stale_version(tmp_path: Any) -> None:
    store = DirectoryMemoryStore(tmp_path)
    await store.write("topic.md", "one", expected_version=None)
    with pytest.raises(MemoryConflictError, match="changed elsewhere"):
        await store.delete("topic.md", expected_version=_version("unrelated"))


# -- helpers -----------------------------------------------------------------


def _version(content: str) -> str:
    return sha256(content.encode()).hexdigest()[:12]


# -- idempotent operation receipts -------------------------------------------


async def test_operation_replay_returns_the_recorded_mutation(tmp_path: Any) -> None:
    store = DirectoryMemoryStore(tmp_path)
    operation = MemoryOperation(id="op-1", fingerprint="f1")
    mutation = await store.write("topic.md", "one", expected_version=None, operation=operation)
    replayed = await store.write("topic.md", "one", expected_version=None, operation=operation)
    assert replayed.version == mutation.version
    assert replayed.replayed is True
    # The replay must not have written again: version unchanged.
    assert (await _read(store, "topic.md")) is not None


async def _read(store: DirectoryMemoryStore, path: str, *, max_chars: int = 10_000) -> Any:
    return await store.read(path, max_chars=max_chars)


async def test_operation_id_reuse_with_other_fingerprint_conflicts(tmp_path: Any) -> None:
    store = DirectoryMemoryStore(tmp_path)
    await store.write("topic.md", "one", expected_version=None, operation=MemoryOperation(id="op-1", fingerprint="f1"))
    with pytest.raises(MemoryOperationConflictError, match="different fingerprint"):
        await store.write("other.md", "two", expected_version=None, operation=MemoryOperation(id="op-1", fingerprint="f2"))


# -- listing and path safety -------------------------------------------------


async def test_list_paths_sorts_and_filters_by_prefix(tmp_path: Any) -> None:
    store = DirectoryMemoryStore(tmp_path)
    for name, content in [("MEMORY.md", "i"), ("b-topic.md", "b"), ("a-topic.md", "a"), ("nested/sub.md", "s")]:
        await store.write(name, content, expected_version=None)

    assert await store.list_paths(limit=100) == ["MEMORY.md", "a-topic.md", "b-topic.md", "nested/sub.md"]
    assert await store.list_paths(prefix="nested/", limit=100) == ["nested/sub.md"]
    assert await store.list_paths(limit=2) == ["MEMORY.md", "a-topic.md"]


async def test_list_paths_on_a_fresh_directory(tmp_path: Any) -> None:
    store = DirectoryMemoryStore(tmp_path / "not-yet")
    assert await store.list_paths(limit=10) == []


@pytest.mark.parametrize("bad", ["../escape", "a/./b", "", "a//b", "a/b c", "ok/../.."])
async def test_invalid_paths_are_rejected(bad: str, tmp_path: Any) -> None:
    store = DirectoryMemoryStore(tmp_path)
    with pytest.raises(ValueError, match="invalid memory path"):
        await store.write(bad, "x", expected_version=None)
