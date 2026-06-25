"""HTTP downloader for direct file URLs."""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from knox.logging import get_logger
from knox.preprocessing.downloaders.base import BaseDownloader
from knox.preprocessing.downloaders.duplicate_detector import DuplicateDetector

logger = get_logger(__name__)

_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

# Content-Type → extension mapping for downloaded files whose URL
# has no recognizable extension (e.g. .aspx, no extension at all).
_CONTENT_TYPE_EXT: dict[str, str] = {
    "text/html": ".html",
    "text/plain": ".txt",
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/epub+zip": ".epub",
}

# FileExtension values the pipeline already recognizes.
_KNOWN_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".doc",
        ".epub",
        ".html",
        ".pptx",
        ".xlsx",
        ".odt",
        ".rst",
        ".rtf",
        ".tex",
        ".md",
        ".txt",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
    }
)


class HTTPDownloader(BaseDownloader):
    """Downloads direct file URLs (PDF, DOCX, etc.) into the sources directory."""

    def __init__(self, timeout: int = 30, detector: DuplicateDetector | None = None) -> None:
        self.timeout = timeout
        self._detector = detector

    def download(self, url: str, target_dir: Path) -> Path | None:
        parsed = urlparse(url)
        if parsed.scheme not in self._ALLOWED_SCHEMES:
            logger.warning("Blocked URL with disallowed scheme: %s", url)
            return None
        target_file, original_name = self._build_target_path(url, target_dir)

        # Check duplicate detector: skip download if URL already known
        if self._detector and self._detector.is_duplicate(url):
            if target_file.exists():
                logger.info("Skipping already-downloaded URL: %s", url)
                return target_file
            # URL was recorded but file was deleted — re-download
            logger.debug("URL %s was recorded but file missing, re-downloading", url)

        logger.debug("Downloading %s -> %s", url, target_file)
        content: bytes | None = None
        content_type: str | None = None
        try:
            request = urllib.request.Request(  # noqa: S310
                url,
                headers={
                    "User-Agent": _USER_AGENT,
                    "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                content = response.read()
                content_type = self._normalize_content_type(response)
        except urllib.error.HTTPError as e:
            logger.debug("HTTP error downloading %s: %s %s", url, e.code, e.reason)
            logger.warning("Skipping %s — HTTP %s", url, e.code)
            return None
        except urllib.error.URLError as e:
            logger.debug("URL error downloading %s: %s", url, e.reason)
            logger.warning("Skipping %s — connection failed", url)
            return None
        except Exception as e:
            logger.debug("Failed to download %s: %s", url, e, exc_info=True)
            logger.warning("Skipping %s — download failed", url)
            return None

        if content is None:  # unreachable — exceptions above all return None
            logger.error("Download returned no content for %s", url)
            return None

        # Pick the right extension based on Content-Type if the URL's
        # original extension is not known to any parser (e.g. .aspx,
        # no extension).
        new_ext = self._pick_extension(original_name, content_type)
        if new_ext != target_file.suffix.lower():
            target_file = target_file.with_suffix(new_ext)
            logger.debug(
                "Re-mapped %s -> %s (Content-Type: %s)",
                original_name,
                target_file.name,
                content_type or "?",
            )

        # Content-based dedup: skip if identical content already exists in target_dir
        content_hash = hashlib.sha256(content).hexdigest()
        for existing_file in target_dir.iterdir():
            if existing_file.is_file() and existing_file.suffix == target_file.suffix and existing_file.read_bytes() == content:
                logger.info(
                    "Skipping %s — identical content already exists as %s",
                    url,
                    existing_file.name,
                )
                if self._detector:
                    self._detector.mark_downloaded(url, content_hash)
                return existing_file

        target_file.write_bytes(content)
        if self._detector:
            self._detector.mark_downloaded(url, content_hash)
        return target_file

    def batch_download(self, urls: list[str], target_dir: Path) -> list[Path]:
        """Download multiple URLs, return paths of successful downloads.

        Args:
            urls: List of URLs to download.
            target_dir: Directory to save downloaded files.

        Returns:
            List of Paths for successfully downloaded files.
        """
        results: list[Path] = []
        for url in urls:
            try:
                path = self.download(url, target_dir)
                if path:
                    results.append(path)
            except Exception as e:
                logger.warning("Failed to download %s: %s", url, e, exc_info=True)
        return results

    def _build_target_path(self, url: str, target_dir: Path) -> tuple[Path, str]:
        """Return (target path, original file name) for a URL."""
        parsed = urlparse(url)
        file_name = Path(parsed.path).name or self._hash_url(url)
        return target_dir / file_name, file_name

    @staticmethod
    def _pick_extension(file_name: str, content_type: str | None) -> str:
        """Return the best file extension for a downloaded file.

        If the existing suffix is already a known parser extension, keep it.
        Otherwise try to map *content_type* to an extension and fall back
        to preserving the original suffix.
        """
        orig_suffix = Path(file_name).suffix.lower()
        if orig_suffix in _KNOWN_EXTENSIONS:
            return orig_suffix
        if content_type:
            mapped = _CONTENT_TYPE_EXT.get(content_type.split(";")[0].strip())
            if mapped:
                return mapped
        return orig_suffix

    @staticmethod
    def _normalize_content_type(response: Any) -> str | None:
        """Extract Content-Type from an HTTP response, returning None on failure."""
        raw = response.headers.get("Content-Type")
        if raw:
            return raw.split(";")[0].strip()
        return None

    @staticmethod
    def _hash_url(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    _ALLOWED_SCHEMES = {"http", "https"}
