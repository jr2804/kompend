"""HTTP downloader for direct file URLs."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from knox.preprocessing.downloaders.base import BaseDownloader
from knox.preprocessing.downloaders.duplicate_detector import DuplicateDetector

logger = logging.getLogger(__name__)

_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"


class HTTPDownloader(BaseDownloader):
    """Downloads direct file URLs (PDF, DOCX, etc.) into the sources directory."""

    def __init__(self, timeout: int = 30, detector: DuplicateDetector | None = None) -> None:
        self.timeout = timeout
        self._detector = detector

    _ALLOWED_SCHEMES = {"http", "https"}

    def download(self, url: str, target_dir: Path) -> Path | None:
        parsed = urlparse(url)
        if parsed.scheme not in self._ALLOWED_SCHEMES:
            logger.warning("Blocked URL with disallowed scheme: %s", url)
            return None
        filename = Path(parsed.path).name or self._hash_url(url)
        target_path = target_dir / filename

        # Check duplicate detector: skip download if URL already known
        if self._detector and self._detector.is_duplicate(url):
            if target_path.exists():
                logger.info("Skipping already-downloaded URL: %s", url)
                return target_path
            # URL was recorded but file was deleted — re-download
            logger.debug("URL %s was recorded but file missing, re-downloading", url)

        logger.debug("Downloading %s -> %s", url, target_path)
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
        except Exception:
            logger.exception("Failed to download %s", url)
            return None

        # Content-based dedup: skip if identical content already exists in target_dir
        content_hash = hashlib.sha256(content).hexdigest()
        for existing_file in target_dir.iterdir():
            if existing_file.is_file() and existing_file.suffix == target_path.suffix and existing_file.read_bytes() == content:
                logger.info(
                    "Skipping %s — identical content already exists as %s",
                    url,
                    existing_file.name,
                )
                if self._detector:
                    self._detector.mark_downloaded(url, content_hash)
                return existing_file

        target_path.write_bytes(content)
        if self._detector:
            self._detector.mark_downloaded(url, content_hash)
        return target_path

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
            except Exception:
                logger.exception("Failed to download %s", url)
        return results

    @staticmethod
    def _hash_url(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:16]
