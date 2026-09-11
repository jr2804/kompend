"""Pytest configuration and fixtures for KNOX tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from knox.rag.zvec_store import EmbeddingProvider

_test_dir = Path(__file__).parent
_data_dir = _test_dir / "data"


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
