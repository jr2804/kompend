"""A :class:`MemoryStore` rooted at a plain host directory.

The harness ships :class:`pydantic_ai_harness.memory.FileStore` for the same
job, but its storage root is a path inside the agent run's sandbox workspace;
the on-machine escape hatch (`workspace=LocalWorkspaceBackend(...)`) is
POSIX-only and raises ``NotImplementedError`` on Windows. Kompend vaults run
on Windows first, so the notebook lives behind this store instead: the same
five-method :class:`pydantic_ai_harness.memory.MemoryStore` protocol, the same
file layout (``MEMORY.md`` plus topic files), plain ``pathlib`` - no sandbox,
no workspace, files exactly where the vault keeps them.

Path rules mirror the harness store contract (segments of ``[A-Za-z0-9_.-]``,
no ``.``/``..``, ``/``-separated) so agent-written paths cannot escape the
root. Search is intentionally NOT implemented: the ``Memory`` capability
falls back to its own bounded scan over :meth:`list_paths`/:meth:`read` for
stores without :class:`SearchableMemoryStore`, which keeps this module down
to the storage contract.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from typing import Final

import anyio.lowlevel
from pydantic_ai_harness.memory import (
    MemoryConflictError,
    MemoryFile,
    MemoryMutation,
    MemoryOperation,
    MemoryOperationConflictError,
)

from kompend.core.config._configs import LLMConfig, LlmMemoryStore

_MAX_RECEIPTS: Final[int] = 1024
"""Bounded replay protection, mirroring the harness stores."""

_VERSION_CHARS: Final[int] = 12

_SEGMENT_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_.-]{1,200}")
"""The harness path-segment rule, restated because it is not exported."""


class DirectoryMemoryStore:
    """A :class:`MemoryStore` keeping Markdown files in one host directory.

    Versions are content fingerprints (``sha256``, cut to
    ``_VERSION_CHARS``), so compare-and-set works across processes. Mutation
    receipts (idempotent-replay protection) are in-process by design: agent
    retries happen inside one run.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._receipts: OrderedDict[str, tuple[str, str | None, bool]] = OrderedDict()

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        target = self._resolve(path)
        data, version = self._load(target)
        if version is None:
            return None
        await anyio.lowlevel.checkpoint()
        content = data.decode("utf-8")
        truncated = len(content) > max_chars
        return MemoryFile(content=content[:max_chars], version=version, operation_id=None, truncated=truncated)

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        return self._receipt(operation)

    async def write(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        replayed = self._receipt(operation)
        if replayed is not None:
            return replayed
        target = self._resolve(path)
        _, current = self._load(target)
        if current != expected_version:
            raise MemoryConflictError(f"memory file {path!r} changed elsewhere: expected {expected_version!r}, found {current!r}")
        await anyio.lowlevel.checkpoint()
        target.parent.mkdir(parents=True, exist_ok=True)
        # Bytes, not text mode: Windows must not translate the agent's \n to \r\n.
        target.write_bytes(content.encode("utf-8"))
        mutation = MemoryMutation(
            version=sha256(content.encode()).hexdigest()[:_VERSION_CHARS],
            replayed=False,
            existed=current is not None,
        )
        self._record(operation, mutation)
        return mutation

    async def delete(
        self,
        path: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        replayed = self._receipt(operation)
        if replayed is not None:
            return replayed
        target = self._resolve(path)
        _, current = self._load(target)
        if current is not None and current != expected_version:
            raise MemoryConflictError(f"memory file {path!r} changed elsewhere: expected {expected_version!r}, found {current!r}")
        await anyio.lowlevel.checkpoint()
        target.unlink(missing_ok=True)
        mutation = MemoryMutation(version=None, replayed=False, existed=current is not None)
        self._record(operation, mutation)
        return mutation

    async def list_paths(self, prefix: str = "", *, limit: int) -> list[str]:
        if prefix:
            _validate_path(prefix.removesuffix("/"))
        await anyio.lowlevel.checkpoint()
        if not self.directory.is_dir():
            return []
        paths = sorted(path.relative_to(self.directory).as_posix() for path in self.directory.rglob("*") if path.is_file())
        return [path for path in paths if path.startswith(prefix)][: max(limit, 0)]

    def _resolve(self, path: str) -> Path:
        _validate_path(path)
        return self.directory.joinpath(*path.split("/"))

    def _receipt(self, operation: MemoryOperation | None) -> MemoryMutation | None:
        if operation is None:
            return None
        stored = self._receipts.get(operation.id)
        if stored is None:
            return None
        fingerprint, version, existed = stored
        if fingerprint != operation.fingerprint:
            raise MemoryOperationConflictError(f"operation id {operation.id!r} reused with a different fingerprint")
        return MemoryMutation(version=version, replayed=True, existed=existed)

    def _record(self, operation: MemoryOperation | None, mutation: MemoryMutation) -> None:
        if operation is None:
            return
        outcome = (operation.fingerprint, mutation.version, mutation.existed)
        self._receipts[operation.id] = outcome
        while len(self._receipts) > _MAX_RECEIPTS:
            self._receipts.popitem(last=False)

    @staticmethod
    def _load(path: Path) -> tuple[bytes, str | None]:
        """The file's bytes and content-version, or ``(b'', None)`` when absent."""
        if not path.is_file():
            return b"", None
        data = path.read_bytes()
        return data, sha256(data).hexdigest()[:_VERSION_CHARS]


def build_notebook_store(llm: LLMConfig | None, memory_dir: Path | None) -> DirectoryMemoryStore | None:
    """The persistent notebook store for a stage, or ``None`` for the in-process one.

    Two config switches decide: ``compile.llm.agent_memory`` turns the notebook
    on and ``compile.llm.memory_store = "vault"`` moves it out of the process
    and into ``memory_dir``. Every stage passes the same directory - the
    harness scopes notebook paths by agent name, so each stage owns its own
    ``<memory_dir>/<agent_name>/MEMORY.md`` and no two stages share a file.
    """
    if llm is None or memory_dir is None or not llm.agent_memory:
        return None
    if llm.memory_store is not LlmMemoryStore.VAULT:
        return None
    return DirectoryMemoryStore(memory_dir)


def _validate_path(path: str) -> None:
    """Reject path strings that could escape the store root (harness contract)."""
    if not path or not all(_SEGMENT_RE.fullmatch(segment) and ".." not in segment and segment != "." for segment in path.split("/")):
        raise ValueError(f"invalid memory path: {path!r}")
