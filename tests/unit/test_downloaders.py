"""Tests for KOMPEND downloaders and duplicate detection."""

from __future__ import annotations

from collections.abc import Generator
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kompend.core.paths import DOWNLOADED_FILE, KOMP_DIR
from kompend.core.store import open_store, vault_db_path
from kompend.core.store.schema import DOWNLOADED_TABLE
from kompend.preprocessing.downloaders.crawler import WebCrawler
from kompend.preprocessing.downloaders.duplicate_detector import DuplicateDetector
from kompend.preprocessing.downloaders.http_downloader import HTTPDownloader

# ---------------------------------------------------------------------------
# HTTPDownloader
# ---------------------------------------------------------------------------


class TestHTTPDownloader:
    """Tests for HTTPDownloader."""

    def test_download_success(self, tmp_path: Path) -> None:
        """Happy path: URL downloads and file is saved."""
        downloader = HTTPDownloader(timeout=5)
        mock_response = self._mock_urlopen()

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_open:
            path = downloader.download("https://example.com/test.pdf", tmp_path)

        assert path is not None
        assert path.name == "test.pdf"
        assert path.parent == tmp_path
        assert path.read_bytes() == b"pdf-content"
        mock_open.assert_called_once()
        call_args = mock_open.call_args
        assert call_args[1]["timeout"] == 5

    def test_download_same_url_twice_skips_second(self, tmp_path: Path) -> None:
        """When same URL is downloaded twice with detector, second is skipped."""
        detector = DuplicateDetector(tmp_path / KOMP_DIR / DOWNLOADED_FILE)
        downloader = HTTPDownloader(timeout=5, detector=detector)

        with patch("urllib.request.urlopen", return_value=self._mock_urlopen()):
            path1 = downloader.download("https://example.com/test.pdf", tmp_path)
            path2 = downloader.download("https://example.com/test.pdf", tmp_path)

        assert path1 is not None
        assert path2 is not None
        assert path1 == path2
        # File should only be written once
        assert path1.read_bytes() == b"pdf-content"

    def test_download_same_url_file_deleted_redownloads(self, tmp_path: Path) -> None:
        """When file was deleted but URL is known, re-download occurs."""
        detector = DuplicateDetector(tmp_path / KOMP_DIR / DOWNLOADED_FILE)
        downloader = HTTPDownloader(timeout=5, detector=detector)

        with patch("urllib.request.urlopen", return_value=self._mock_urlopen()):
            path1 = downloader.download("https://example.com/test.pdf", tmp_path)

        assert path1 is not None
        path1.unlink()

        with patch("urllib.request.urlopen", return_value=self._mock_urlopen()):
            path2 = downloader.download("https://example.com/test.pdf", tmp_path)

        assert path2 is not None
        assert path2.exists()

    def test_download_content_dedup_same_content(self, tmp_path: Path) -> None:
        """When different URL produces same content, skip download."""
        downloader = HTTPDownloader(timeout=5)
        # Pre-create file with same content
        existing = tmp_path / "other.pdf"
        existing.write_bytes(b"pdf-content")

        with patch("urllib.request.urlopen", return_value=self._mock_urlopen()):
            path = downloader.download("https://example.com/test.pdf", tmp_path)

        assert path is not None
        assert path == existing  # Returns existing file, not new one

    def test_download_different_content_both_saved(self, tmp_path: Path) -> None:
        """Different content from different URLs are both saved."""
        downloader = HTTPDownloader(timeout=5)

        def _side_effect(request: Any, timeout: int | None = None) -> Any:
            response = MagicMock()
            if "a.pdf" in request.full_url:
                response.read.return_value = b"content-a"
            else:
                response.read.return_value = b"content-b"
            response.__enter__ = lambda s: response
            response.__exit__ = MagicMock(return_value=False)
            return response

        with patch("urllib.request.urlopen", side_effect=_side_effect):
            path1 = downloader.download("https://example.com/a.pdf", tmp_path)
            path2 = downloader.download("https://example.com/b.pdf", tmp_path)

        assert path1 is not None
        assert path2 is not None
        assert path1 != path2
        assert path1.read_bytes() == b"content-a"
        assert path2.read_bytes() == b"content-b"

    def test_download_no_detector_still_works(self, tmp_path: Path) -> None:
        """Downloader works without detector (backward compatible)."""
        downloader = HTTPDownloader(timeout=5)

        with patch("urllib.request.urlopen", return_value=self._mock_urlopen()):
            path = downloader.download("https://example.com/test.pdf", tmp_path)

        assert path is not None
        assert path.name == "test.pdf"

    def test_download_failure_returns_none(self, tmp_path: Path) -> None:
        """Network error returns None without raising."""
        downloader = HTTPDownloader(timeout=1)

        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            path = downloader.download("https://example.com/bad.pdf", tmp_path)

        assert path is None

    def test_download_no_filename_in_url(self, tmp_path: Path) -> None:
        """URL without a path uses hash instead."""
        downloader = HTTPDownloader(timeout=5)

        with patch("urllib.request.urlopen", return_value=self._mock_urlopen()):
            path = downloader.download("https://example.com", tmp_path)

        assert path is not None
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

        with patch("urllib.request.urlopen", return_value=self._mock_urlopen()):
            results = downloader.batch_download(urls, tmp_path)

        assert len(results) == 3
        for path in results:
            assert path.parent == tmp_path

    def test_batch_download_mixed_success_failure(self, tmp_path: Path) -> None:
        """batch_download handles mixed success/failure without raising."""
        downloader = HTTPDownloader(timeout=5)

        def _side_effect(request: Any, timeout: int | None = None) -> Any:
            if "fail" in request.full_url:
                raise OSError("failure")
            return self._mock_urlopen()

        urls = [
            "https://example.com/good.pdf",
            "https://example.com/fail.pdf",
            "https://example.com/ok.pdf",
        ]

        with patch("urllib.request.urlopen", side_effect=_side_effect):
            results = downloader.batch_download(urls, tmp_path)

        assert len(results) == 2
        assert all("fail" not in str(p) for p in results)

    def test_download_aspx_saved_as_html_from_content_type(self, tmp_path: Path) -> None:
        """A URL ending in .aspx with text/html Content-Type is saved as .html."""
        downloader = HTTPDownloader(timeout=5)
        mock_response = self._mock_urlopen(data=b"<html>hi</html>", content_type="text/html")

        with patch("urllib.request.urlopen", return_value=mock_response):
            path = downloader.download("https://example.com/page.aspx", tmp_path)

        assert path is not None
        assert path.suffix == ".html"
        assert path.stem == "page"
        assert path.read_bytes() == b"<html>hi</html>"

    def test_download_no_extension_saved_as_html(self, tmp_path: Path) -> None:
        """A URL with no file extension and text/html Content-Type is saved as .html."""
        downloader = HTTPDownloader(timeout=5)
        mock_response = self._mock_urlopen(data=b"<html>hi</html>", content_type="text/html")

        with patch("urllib.request.urlopen", return_value=mock_response):
            path = downloader.download("https://example.com/20q-listening-effort-29239", tmp_path)

        assert path is not None
        assert path.suffix == ".html"
        assert path.stem == "20q-listening-effort-29239"
        assert path.read_bytes() == b"<html>hi</html>"

    def test_download_pdf_extension_kept_for_text_html(self, tmp_path: Path) -> None:
        """A URL with a known extension (.pdf) keeps it regardless of Content-Type."""
        downloader = HTTPDownloader(timeout=5)
        mock_response = self._mock_urlopen(data=b"fake", content_type="text/html")

        with patch("urllib.request.urlopen", return_value=mock_response):
            path = downloader.download("https://example.com/doc.pdf", tmp_path)

        assert path is not None
        assert path.suffix == ".pdf"

    def test_download_unknown_content_type_keeps_original_ext(self, tmp_path: Path) -> None:
        """An unrecognized Content-Type keeps the original extension."""
        downloader = HTTPDownloader(timeout=5)
        mock_response = self._mock_urlopen(data=b"foo", content_type="application/octet-stream")

        with patch("urllib.request.urlopen", return_value=mock_response):
            path = downloader.download("https://example.com/data.xyz", tmp_path)

        assert path is not None
        assert path.suffix == ".xyz"

    def test_download_missing_content_type_keeps_original_ext(self, tmp_path: Path) -> None:
        """A missing Content-Type header keeps the original extension."""
        downloader = HTTPDownloader(timeout=5)
        mock_response = self._mock_urlopen(data=b"foo", content_type="")
        mock_response.headers = {}

        with patch("urllib.request.urlopen", return_value=mock_response):
            path = downloader.download("https://example.com/page.aspx", tmp_path)

        assert path is not None
        assert path.suffix == ".aspx"

    @staticmethod
    def _mock_urlopen(data: bytes = b"pdf-content", content_type: str = "application/pdf") -> MagicMock:
        """Create a mock context manager for urllib.request.urlopen."""
        response = MagicMock()
        response.read.return_value = data
        response.headers = {"Content-Type": content_type}
        response.__enter__ = lambda s: response
        response.__exit__ = MagicMock(return_value=False)
        return response


# ---------------------------------------------------------------------------
# WebCrawler
# ---------------------------------------------------------------------------


class TestWebCrawler:
    """Tests for WebCrawler."""

    def test_crawl4ai_not_available_returns_none(self, tmp_path: Path) -> None:
        """When crawl4ai is missing, download returns None."""
        # crawl4ai is a hard dep, but we can simulate network failure
        crawler = WebCrawler()
        crawler.download("https://example.com", tmp_path)

    def test_download_success(self, tmp_path: Path) -> None:
        """Happy path: crawl4ai returns markdown and file is saved."""
        pytest.importorskip("crawl4ai")

        mock_result = MagicMock()
        mock_result.markdown = "# Hello\n\nThis is markdown."

        async def _fake_arun(_url: str) -> MagicMock:  # noqa: ANN401
            return mock_result

        with patch(
            "kompend.preprocessing.downloaders.crawler.AsyncWebCrawler",
        ) as mock_crawler_cls:
            mock_instance = MagicMock()
            mock_instance.arun.side_effect = _fake_arun
            mock_crawler_cls.return_value.__aenter__.return_value = mock_instance

            crawler = WebCrawler()
            result = crawler.download("https://docs.example.com", tmp_path)

        assert result is not None
        assert result.name == "docs.example.com.md"
        assert result.read_text(encoding="utf-8") == "# Hello\n\nThis is markdown."

    def test_download_same_url_skips_with_detector(self, tmp_path: Path) -> None:
        """When same URL is crawled twice with detector, second is skipped."""
        pytest.importorskip("crawl4ai")

        detector = DuplicateDetector(tmp_path / KOMP_DIR / DOWNLOADED_FILE)
        mock_result = MagicMock()
        mock_result.markdown = "# Hello\n\nThis is markdown."

        async def _fake_arun(_url: str) -> MagicMock:  # noqa: ANN401
            return mock_result

        with patch("kompend.preprocessing.downloaders.crawler.AsyncWebCrawler") as mock_crawler_cls:
            mock_instance = MagicMock()
            mock_instance.arun.side_effect = _fake_arun
            mock_crawler_cls.return_value.__aenter__.return_value = mock_instance

            crawler = WebCrawler(detector=detector)
            result1 = crawler.download("https://docs.example.com", tmp_path)
            result2 = crawler.download("https://docs.example.com", tmp_path)

        assert result1 is not None
        assert result2 is not None
        assert result1 == result2

    def test_download_no_markdown_returns_none(self, tmp_path: Path) -> None:
        """When crawl4ai returns no markdown, download returns None."""
        pytest.importorskip("crawl4ai")

        mock_result = MagicMock()
        mock_result.markdown = None

        async def _fake_arun(_url: str) -> MagicMock:  # noqa: ANN401
            return mock_result

        with patch(
            "kompend.preprocessing.downloaders.crawler.AsyncWebCrawler",
        ) as mock_crawler_cls:
            mock_instance = MagicMock()
            mock_instance.arun.side_effect = _fake_arun
            mock_crawler_cls.return_value.__aenter__.return_value = mock_instance

            crawler = WebCrawler()
            result = crawler.download("https://docs.example.com", tmp_path)

        assert result is None

    def test_batch_download(self, tmp_path: Path) -> None:
        """batch_download processes multiple URLs."""
        pytest.importorskip("crawl4ai")

        mock_result = MagicMock()
        mock_result.markdown = "# content"

        async def _fake_arun(_url: str) -> MagicMock:  # noqa: ANN401
            return mock_result

        with patch(
            "kompend.preprocessing.downloaders.crawler.AsyncWebCrawler",
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

    def test_stealth_always_enabled(self, tmp_path: Path) -> None:
        """Stealth is always on the BrowserConfig (undetected=False path)."""
        pytest.importorskip("crawl4ai")

        captured: list[Any] = []

        class _FakeAsyncWebCrawler:
            def __init__(self, config: Any = None, **kwargs: Any) -> None:
                captured.append((config, kwargs))

            async def __aenter__(self) -> _FakeAsyncWebCrawler:
                return self

            async def __aexit__(self, *_args: Any) -> None:
                pass

            async def arun(self, _url: str) -> MagicMock:
                m = MagicMock()
                m.markdown = "# x"
                m.success = True
                return m

        with patch(
            "kompend.preprocessing.downloaders.crawler.AsyncWebCrawler",
            _FakeAsyncWebCrawler,
        ):
            # undetected=False -> stealth fallback path
            WebCrawler(undetected=False).download("https://docs.example.com", tmp_path)

        assert len(captured) == 1
        assert captured[0][0].enable_stealth is True
        assert captured[0][0].headless is True
        assert "crawler_strategy" not in captured[0][1]

    def test_undetected_passes_strategy(self, tmp_path: Path) -> None:
        """Undetected mode passes a crawler_strategy + channel to AsyncWebCrawler."""
        pytest.importorskip("crawl4ai")

        captured: list[Any] = []

        class _FakeAsyncWebCrawler:
            def __init__(self, config: Any = None, **kwargs: Any) -> None:
                captured.append((config, kwargs))

            async def __aenter__(self) -> _FakeAsyncWebCrawler:
                return self

            async def __aexit__(self, *_args: Any) -> None:
                pass

            async def arun(self, _url: str) -> MagicMock:
                m = MagicMock()
                m.markdown = "# x"
                m.success = True
                return m

        with (
            patch(
                "kompend.preprocessing.downloaders.crawler.AsyncWebCrawler",
                _FakeAsyncWebCrawler,
            ),
            patch(
                "kompend.preprocessing.downloaders.crawler._detect_real_browser_channel",
                return_value="msedge",
            ),
        ):
            WebCrawler(undetected=True).download("https://docs.example.com", tmp_path)

        assert len(captured) == 1
        assert "crawler_strategy" in captured[0][1]
        assert captured[0][0].chrome_channel == "msedge"

    def test_undetected_falls_back_when_no_browser(self, tmp_path: Path) -> None:
        """Undetected falls back to stealth when no real browser is found."""
        pytest.importorskip("crawl4ai")

        captured: list[Any] = []

        class _FakeAsyncWebCrawler:
            def __init__(self, config: Any = None, **kwargs: Any) -> None:
                captured.append((config, kwargs))

            async def __aenter__(self) -> _FakeAsyncWebCrawler:
                return self

            async def __aexit__(self, *_args: Any) -> None:
                pass

            async def arun(self, _url: str) -> MagicMock:
                m = MagicMock()
                m.markdown = "# x"
                m.success = True
                return m

        with patch(
            "kompend.preprocessing.downloaders.crawler.AsyncWebCrawler",
            _FakeAsyncWebCrawler,
        ):
            # _detect_real_browser_channel returns None via autouse fixture
            WebCrawler(undetected=True).download("https://docs.example.com", tmp_path)

        assert len(captured) == 1
        assert "crawler_strategy" not in captured[0][1]

    @pytest.fixture(autouse=True)
    def _fake_chromium(self) -> Generator[None]:
        """Make WebCrawler think Chromium is installed and no real browser
        is present (stealth fallback) in every test, unless overridden.
        """
        with (
            patch("kompend.preprocessing.downloaders.crawler._chromium_available", return_value=True),
            patch("kompend.preprocessing.downloaders.crawler._detect_real_browser_channel", return_value=None),
        ):
            yield


# ---------------------------------------------------------------------------
# DuplicateDetector
# ---------------------------------------------------------------------------


class TestDuplicateDetector:
    """Tests for DuplicateDetector (no network needed)."""

    def test_init_empty_writes_no_file(self, tmp_path: Path) -> None:
        """A newly initialized detector is empty and creates no legacy file."""
        detector = self._detector(tmp_path)

        assert not detector.is_duplicate("https://example.com")
        # Clean cut: the legacy flat file is never written.
        assert not (tmp_path / KOMP_DIR / DOWNLOADED_FILE).exists()

    def test_mark_and_check(self, tmp_path: Path) -> None:
        """Marking a URL makes it detectable as duplicate."""
        detector = self._detector(tmp_path)

        assert not detector.is_duplicate("https://example.com/doc.pdf")

        detector.mark_downloaded("https://example.com/doc.pdf", "abc123")
        assert detector.is_duplicate("https://example.com/doc.pdf")
        assert not detector.is_duplicate("https://example.com/other.pdf")

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        """The database persists between detector instances."""
        d1 = self._detector(tmp_path)
        d1.mark_downloaded("https://example.com/a.pdf", "hash1")
        d1.mark_downloaded("https://example.com/b.pdf", "hash2")

        d2 = self._detector(tmp_path)
        assert d2.is_duplicate("https://example.com/a.pdf")
        assert d2.is_duplicate("https://example.com/b.pdf")
        assert not d2.is_duplicate("https://example.com/c.pdf")

    def test_row_written_to_store(self, tmp_path: Path) -> None:
        """mark_downloaded lands a row in the ``downloaded`` table."""
        detector = self._detector(tmp_path)
        detector.mark_downloaded("https://example.com/doc.pdf", "abc123")

        with open_store(vault_db_path(tmp_path)) as conn:
            rows = {row["url"]: row["sha256"] for row in conn.execute(f"SELECT url, sha256 FROM {DOWNLOADED_TABLE}").fetchall()}  # noqa: S608
        assert rows == {"https://example.com/doc.pdf": "abc123"}

    def test_legacy_file_ignored(self, tmp_path: Path) -> None:
        """A leftover flat ``downloaded.json`` is not read (clean cut)."""
        legacy = tmp_path / KOMP_DIR / DOWNLOADED_FILE
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text('{"https://example.com/old": "oldhash"}', encoding="utf-8")

        detector = self._detector(tmp_path)
        assert not detector.is_duplicate("https://example.com/old")

    def test_batch_check(self, tmp_path: Path) -> None:
        """batch_check returns correct mapping for multiple URLs."""
        detector = self._detector(tmp_path)
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

    def test_compute_hash(self) -> None:
        """compute_hash returns the correct SHA256 hex digest."""
        content = b"hello world"
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert DuplicateDetector.compute_hash(content) == expected

    @staticmethod
    def _detector(tmp_path: Path) -> DuplicateDetector:
        return DuplicateDetector(tmp_path / KOMP_DIR / DOWNLOADED_FILE)
