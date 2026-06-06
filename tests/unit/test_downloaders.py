"""Tests for KNOX downloaders and duplicate detection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knox.preprocessing.downloaders.duplicate_detector import DuplicateDetector
from knox.preprocessing.downloaders.http_downloader import HTTPDownloader
from knox.preprocessing.downloaders.crawler import CRAWL4AI_AVAILABLE, WebCrawler


# ---------------------------------------------------------------------------
# HTTPDownloader
# ---------------------------------------------------------------------------

class TestHTTPDownloader:
    """Tests for HTTPDownloader."""

    def test_download_success(self, tmp_path: Path) -> None:
        """Happy path: URL downloads and file is saved."""
        downloader = HTTPDownloader(timeout=5)

        with patch("urllib.request.urlretrieve") as mock_retrieve:
            path = downloader.download("https://example.com/test.pdf", tmp_path)

        assert path is not None
        assert path.name == "test.pdf"
        assert path.parent == tmp_path
        mock_retrieve.assert_called_once_with(
            "https://example.com/test.pdf",
            str(path),
        )

    def test_download_collision_safe_filename(self, tmp_path: Path) -> None:
        """When file exists, append a counter suffix."""
        # Create the file beforehand so the downloader must increment
        existing = tmp_path / "test.pdf"
        existing.write_text("dummy")

        downloader = HTTPDownloader(timeout=5)

        with patch("urllib.request.urlretrieve"):
            path = downloader.download("https://example.com/test.pdf", tmp_path)

        assert path is not None
        assert path.name == "test_1.pdf"

    def test_download_failure_returns_none(self, tmp_path: Path) -> None:
        """Network error returns None without raising."""
        downloader = HTTPDownloader(timeout=1)

        with patch("urllib.request.urlretrieve", side_effect=OSError("timeout")):
            path = downloader.download("https://example.com/bad.pdf", tmp_path)

        assert path is None

    def test_download_no_filename_in_url(self, tmp_path: Path) -> None:
        """URL without a path uses hash instead."""
        downloader = HTTPDownloader(timeout=5)

        with patch("urllib.request.urlretrieve"):
            path = downloader.download("https://example.com", tmp_path)

        assert path is not None
        # Name should be the first 16 chars of the SHA256 of the URL
        from hashlib import sha256
        expected = sha256(b"https://example.com").hexdigest()[:16]
        assert path.stem == expected

    def test_batch_download_all_succeed(self, tmp_path: Path) -> None:
        """batch_download returns paths for all successful downloads."""
        downloader = HTTPDownloader(timeout=5)
        urls = [
            "https://example.com/a.pdf",
            "https://example.com/b.pdf",
            "https://example.com/c.pdf",
        ]

        with patch("urllib.request.urlretrieve"):
            results = downloader.batch_download(urls, tmp_path)

        assert len(results) == 3
        for path in results:
            assert path.parent == tmp_path

    def test_batch_download_mixed_success_failure(self, tmp_path: Path) -> None:
        """batch_download handles mixed success/failure without raising."""
        downloader = HTTPDownloader(timeout=5)

        def _side_effect(url: str, _: str) -> None:
            if "fail" in url:
                raise OSError("failure")
            return None

        urls = [
            "https://example.com/good.pdf",
            "https://example.com/fail.pdf",
            "https://example.com/ok.pdf",
        ]

        with patch("urllib.request.urlretrieve", side_effect=_side_effect):
            results = downloader.batch_download(urls, tmp_path)

        assert len(results) == 2
        assert all("fail" not in str(p) for p in results)


# ---------------------------------------------------------------------------
# WebCrawler
# ---------------------------------------------------------------------------

class TestWebCrawler:
    """Tests for WebCrawler."""

    def test_crawl4ai_not_available_returns_none(self, tmp_path: Path) -> None:
        """When crawl4ai is missing, download returns None."""
        # Temporarily override the flag
        with patch("knox.preprocessing.downloaders.crawler.CRAWL4AI_AVAILABLE", False):
            crawler = WebCrawler()
            result = crawler.download("https://example.com", tmp_path)

        assert result is None

    def test_download_success(self, tmp_path: Path) -> None:
        """Happy path: crawl4ai returns markdown and file is saved."""
        if not CRAWL4AI_AVAILABLE:
            pytest.skip("crawl4ai not installed")

        mock_result = MagicMock()
        mock_result.markdown = "# Hello\n\nThis is markdown."

        async def _fake_arun(_url: str) -> MagicMock:  # noqa: ANN401
            return mock_result

        with patch(
            "knox.preprocessing.downloaders.crawler.AsyncWebCrawler",
        ) as mock_crawler_cls:
            mock_instance = MagicMock()
            mock_instance.arun.side_effect = _fake_arun
            mock_crawler_cls.return_value.__aenter__.return_value = mock_instance

            crawler = WebCrawler()
            result = crawler.download("https://docs.example.com", tmp_path)

        assert result is not None
        assert result.name == "docs.example.com.md"
        assert result.read_text(encoding="utf-8") == "# Hello\n\nThis is markdown."

    def test_download_no_markdown_returns_none(self, tmp_path: Path) -> None:
        """When crawl4ai returns no markdown, download returns None."""
        if not CRAWL4AI_AVAILABLE:
            pytest.skip("crawl4ai not installed")

        mock_result = MagicMock()
        mock_result.markdown = None

        async def _fake_arun(_url: str) -> MagicMock:  # noqa: ANN401
            return mock_result

        with patch(
            "knox.preprocessing.downloaders.crawler.AsyncWebCrawler",
        ) as mock_crawler_cls:
            mock_instance = MagicMock()
            mock_instance.arun.side_effect = _fake_arun
            mock_crawler_cls.return_value.__aenter__.return_value = mock_instance

            crawler = WebCrawler()
            result = crawler.download("https://docs.example.com", tmp_path)

        assert result is None

    def test_download_sync_alias(self, tmp_path: Path) -> None:
        """download_sync should behave identical to download()."""
        with patch("knox.preprocessing.downloaders.crawler.CRAWL4AI_AVAILABLE", False):
            crawler = WebCrawler()
            result = crawler.download_sync("https://example.com", tmp_path)

        assert result is None

    def test_batch_download(self, tmp_path: Path) -> None:
        """batch_download processes multiple URLs."""
        if not CRAWL4AI_AVAILABLE:
            pytest.skip("crawl4ai not installed")

        mock_result = MagicMock()
        mock_result.markdown = "# content"

        async def _fake_arun(_url: str) -> MagicMock:  # noqa: ANN401
            return mock_result

        with patch(
            "knox.preprocessing.downloaders.crawler.AsyncWebCrawler",
        ) as mock_crawler_cls:
            mock_instance = MagicMock()
            mock_instance.arun.side_effect = _fake_arun
            mock_crawler_cls.return_value.__aenter__.return_value = mock_instance

            crawler = WebCrawler()
            urls = [
                "https://docs.example.com/a",
                "https://docs.example.com/b",
            ]
            results = crawler.batch_download(urls, tmp_path)

        assert len(results) == 2


# ---------------------------------------------------------------------------
# DuplicateDetector
# ---------------------------------------------------------------------------

class TestDuplicateDetector:
    """Tests for DuplicateDetector (no network needed)."""

    def test_init_creates_empty_db(self, tmp_path: Path) -> None:
        """A newly initialized detector has an empty database."""
        db_path = tmp_path / "hashes.json"
        detector = DuplicateDetector(db_path)

        assert not db_path.exists() or db_path.read_text() == "{}"
        assert not detector.is_duplicate("https://example.com")

    def test_mark_and_check(self, tmp_path: Path) -> None:
        """Marking a URL makes it detectable as duplicate."""
        db_path = tmp_path / "hashes.json"
        detector = DuplicateDetector(db_path)

        assert not detector.is_duplicate("https://example.com/doc.pdf")

        detector.mark_downloaded("https://example.com/doc.pdf", "abc123")
        assert detector.is_duplicate("https://example.com/doc.pdf")
        assert not detector.is_duplicate("https://example.com/other.pdf")

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        """The database persists between detector instances."""
        db_path = tmp_path / "hashes.json"

        d1 = DuplicateDetector(db_path)
        d1.mark_downloaded("https://example.com/a.pdf", "hash1")
        d1.mark_downloaded("https://example.com/b.pdf", "hash2")

        d2 = DuplicateDetector(db_path)
        assert d2.is_duplicate("https://example.com/a.pdf")
        assert d2.is_duplicate("https://example.com/b.pdf")
        assert not d2.is_duplicate("https://example.com/c.pdf")

    def test_batch_check(self, tmp_path: Path) -> None:
        """batch_check returns correct mapping for multiple URLs."""
        db_path = tmp_path / "hashes.json"
        detector = DuplicateDetector(db_path)
        detector.mark_downloaded("https://example.com/known.pdf", "hash")

        urls = [
            "https://example.com/known.pdf",
            "https://example.com/unknown.pdf",
            "https://example.com/also_unknown.pdf",
        ]
        result = detector.batch_check(urls)

        assert result == {
            "https://example.com/known.pdf": True,
            "https://example.com/unknown.pdf": False,
            "https://example.com/also_unknown.pdf": False,
        }

    def test_corrupt_db_starts_fresh(self, tmp_path: Path) -> None:
        """A corrupt JSON file is handled gracefully."""
        db_path = tmp_path / "hashes.json"
        db_path.write_text("not valid json{{{", encoding="utf-8")

        detector = DuplicateDetector(db_path)
        assert not detector.is_duplicate("https://example.com")
        # Writing should work despite initial corruption
        detector.mark_downloaded("https://example.com", "hash")
        assert detector.is_duplicate("https://example.com")

    def test_compute_hash(self) -> None:
        """compute_hash returns the correct SHA256 hex digest."""
        content = b"hello world"
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert DuplicateDetector.compute_hash(content) == expected

    def test_db_path_creates_parent_dirs(self, tmp_path: Path) -> None:
        """mark_downloaded creates parent directories if needed."""
        db_path = tmp_path / "subdir" / "nested" / "hashes.json"
        detector = DuplicateDetector(db_path)
        detector.mark_downloaded("https://example.com", "hash")
        assert db_path.exists()
        data = json.loads(db_path.read_text(encoding="utf-8"))
        assert data == {"https://example.com": "hash"}