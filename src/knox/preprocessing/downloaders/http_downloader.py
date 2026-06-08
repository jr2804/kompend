"""HTTP downloader for direct file URLs."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from knox.preprocessing.downloaders.base import BaseDownloader

logger = logging.getLogger(__name__)

_USER_AGENT = "KNOX/0.1 (Knowledge Organizer for eXtended Documents)"


class HTTPDownloader(BaseDownloader):
    """Downloads direct file URLs (PDF, DOCX, etc.) into the sources directory."""

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    _ALLOWED_SCHEMES = {"http", "https"}

    def download(self, url: str, target_dir: Path) -> Path | None:
        parsed = urlparse(url)
        if parsed.scheme not in self._ALLOWED_SCHEMES:
            logger.warning("Blocked URL with disallowed scheme: %s", url)
            return None
        filename = Path(parsed.path).name or self._hash_url(url)
        target_path = target_dir / filename

        # Collision-safe filename
        counter = 1
        original_target = target_path
        while target_path.exists():
            stem = original_target.stem
            suffix = original_target.suffix
            target_path = target_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        logger.info("Downloading %s -> %s", url, target_path)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                target_path.write_bytes(response.read())
        except Exception:
            logger.exception("Failed to download %s", url)
            return None
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
                logger.exception(f"Failed to download {url}")
        return results

    @staticmethod
    def _hash_url(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:16]
