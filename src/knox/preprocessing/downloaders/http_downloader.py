"""HTTP downloader for direct file URLs."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from knox.preprocessing.downloaders.base import BaseDownloader

logger = logging.getLogger(__name__)


class HTTPDownloader(BaseDownloader):
    """Downloads direct file URLs (PDF, DOCX, etc.) into the sources directory."""

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    def download(self, url: str, target_dir: Path) -> Path | None:
        parsed = urlparse(url)
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
            urllib.request.urlretrieve(url, str(target_path))
        except Exception:
            logger.exception("Failed to download %s", url)
            return None
        return target_path

    @staticmethod
    def _hash_url(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:16]
