"""Pytest configuration, fixtures, and shared narrowing helpers for KNOX tests.

``as_obj`` / ``as_list`` exist because KNOX's JSON types are precise recursive
unions on purpose (see ``knox/types.py``): a payload read from LLM output, a
config file or a cache must be narrowed before use rather than silently
propagating ``Any``. In tests the expected shape is known, so narrowing is an
assertion — these helpers fail loudly if a payload is not what the test claims.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from knox.rag.zvec_store import EmbeddingProvider

_test_dir = Path(__file__).parent
_data_dir = _test_dir / "data"


def as_obj(value: object) -> dict[str, Any]:
    """Narrow *value* to a JSON object, asserting it is one."""
    assert isinstance(value, dict), f"expected a JSON object, got {type(value).__name__}"
    return value


def as_list(value: object) -> list[Any]:
    """Narrow *value* to a JSON array, asserting it is one."""
    assert isinstance(value, list), f"expected a JSON array, got {type(value).__name__}"
    return value


def as_float(value: object) -> float:
    """Narrow *value* to a number, asserting it is one."""
    assert isinstance(value, (int, float)), f"expected a number, got {type(value).__name__}"
    return float(value)


def as_str(value: object) -> str:
    """Narrow *value* to a string, asserting it is one."""
    assert isinstance(value, str), f"expected a string, got {type(value).__name__}"
    return value


class DeterministicTestEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based embedding for tests only.

    NOT semantically meaningful — same text always produces the same vector.
    Lives in test code, never in production. Replaces the old HashEmbeddingProvider.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(digest[i % 32] - 128) / 128.0 for i in range(self._dim)]
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm > 0 else vec


def make_tmp_vault() -> Path:
    """Return an absolute throwaway vault root.

    Store-backed ``PipelineConfig``s must not carry a CWD-relative
    ``source_dir``/``output_dir`` — that would resolve the vault DB against the
    process CWD and scatter ``.knox/`` state into the repository root.
    """
    return Path(tempfile.mkdtemp(prefix="knox-vault-"))


@pytest.fixture(scope="session", autouse=True)
def guard_repo_root_vault() -> Iterator[None]:
    """Fail if the suite scatters ``.knox/`` state into the repository root.

    A store-backed config (or store) handed a CWD-relative path resolves its DB
    against the process CWD and writes ``<repo>/.knox/knox.db``. That happened
    once (a ``PipelineConfig`` with ``source_dir=Path(".")``); this guard keeps
    it fixed. ``make_tmp_vault()`` is the correct root for any such config.
    """
    stray = _test_dir.parent / ".knox"
    pre_existing = stray.exists()
    yield
    if not pre_existing and stray.exists():
        msg = (
            f"The test suite created {stray} — a store-backed config got a CWD-relative "
            f"path. Use make_tmp_vault() for PipelineConfig source_dir/output_dir "
            f"(and any other vault root handed to a store)."
        )
        raise AssertionError(msg)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Return the path to the test data directory."""
    return _data_dir


@pytest.fixture(scope="session")
def source_pdfs(sources_dir: Path) -> list[Path]:
    """Return sorted list of PDF files in the test corpus."""
    if not sources_dir.exists():
        return []
    return sorted(sources_dir.glob("*.pdf"))


@pytest.fixture(scope="session")
def sources_dir() -> Path:
    """Return the path to ``tests/data/sources/`` (local PDF/DOCX samples)."""
    return _data_dir / "sources"


@pytest.fixture(scope="session")
def test_urls() -> list[str]:
    """Return URLs from ``tests/data/sources/urls.txt`` (one per line, ``#`` comments)."""
    urls_file = _data_dir / "sources" / "urls.txt"
    if not urls_file.exists():
        return []
    lines = urls_file.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


@pytest.fixture
def example_wiki_cache(request: pytest.FixtureRequest) -> Path:
    """Create and return the ``example-wiki/`` cache subdirectory.

    Uses ``request.config.cache.mkdir()`` — the official pytest Cache API.
    The cache root is configured via ``cache_dir`` in ``pyproject.toml``.
    """
    return Path(str(request.config.cache.mkdir("example-wiki")))
