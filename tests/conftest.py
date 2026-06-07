"""Pytest configuration and fixtures for KNOX tests."""

from __future__ import annotations

from pathlib import Path

import pytest

_test_dir = Path(__file__).parent
_data_dir = _test_dir / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Return the path to the test data directory."""
    return _data_dir


@pytest.fixture(scope="session")
def sources_dir() -> Path:
    """Return the path to ``tests/data/sources/`` (local PDF/DOCX samples)."""
    return _data_dir / "sources"


@pytest.fixture(scope="session")
def source_pdfs(sources_dir: Path) -> list[Path]:
    """Return sorted list of PDF files in the test corpus."""
    if not sources_dir.exists():
        return []
    return sorted(sources_dir.glob("*.pdf"))


@pytest.fixture(scope="session")
def test_urls() -> list[str]:
    """Return URLs from ``tests/data/urls.txt`` (one per line, ``#`` comments)."""
    urls_file = _data_dir / "urls.txt"
    if not urls_file.exists():
        return []
    lines = urls_file.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


@pytest.fixture(scope="session")
def cache_subdir(request: pytest.FixtureRequest, subdir: str) -> Path:
    """Return a subdirectory in the pytest cache directory."""
    return Path(request.config.cache.mkdir(subdir))
