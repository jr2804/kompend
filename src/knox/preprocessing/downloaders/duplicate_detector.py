"""SHA256-based duplicate detection for downloaded URLs.

Uses a simple JSON file as storage to track which URLs have been
downloaded and their corresponding content hashes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """Tracks downloaded URLs and content hashes to detect duplicates.

    Uses a JSON file as persistent storage. Each entry maps a URL to
    a SHA256 hash of its content.
    """

    def __init__(self, db_file: Path) -> None:
        """Initialize the detector with a path to the JSON database.

        Args:
            db_file: Path to the JSON file used for persistent storage.
                     Will be created if it doesn't exist.
        """
        self.db_file = db_file
        self._db: dict[str, str] = {}
        self._load()

    def is_duplicate(self, url: str) -> bool:
        """Check if a URL has been downloaded before.

        Args:
            url: The URL to check.

        Returns:
            True if the URL is already in the database, False otherwise.
        """
        return url in self._db

    def mark_downloaded(self, url: str, sha256: str) -> None:
        """Record a URL as downloaded with its content hash.

        Args:
            url: The URL that was downloaded.
            sha256: The SHA256 hex digest of the downloaded content.
        """
        self._db[url] = sha256
        self._save()

    def batch_check(self, urls: list[str]) -> dict[str, bool]:
        """Check multiple URLs at once.

        Args:
            urls: List of URLs to check.

        Returns:
            Dictionary mapping each URL to whether it is a duplicate.
        """
        return {url: self.is_duplicate(url) for url in urls}

    @staticmethod
    def compute_hash(content: bytes) -> str:
        """Compute the SHA256 hex digest of content.

        Args:
            content: The raw bytes to hash.

        Returns:
            SHA256 hex digest string.
        """
        return hashlib.sha256(content).hexdigest()

    def _load(self) -> None:
        """Load the database from disk."""
        if self.db_file.exists():
            try:
                raw = self.db_file.read_text(encoding="utf-8")
                self._db = json.loads(raw)
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt or unreadable DB at %s, starting fresh", self.db_file, exc_info=True)
                self._db = {}
        else:
            self._db = {}

    def _save(self) -> None:
        """Persist the database to disk."""
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self.db_file.write_text(
            json.dumps(self._db, indent=2, sort_keys=True),
            encoding="utf-8",
        )
