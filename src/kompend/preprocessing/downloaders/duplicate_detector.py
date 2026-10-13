"""SHA256-based duplicate detection for downloaded URLs.

The URL → content-hash map lives in the ``downloaded`` table of the single
``.kompend/kompend.db`` store (was the flat ``cache/downloaded.json`` file).
"""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants;
# no runtime value ever enters an identifier or statement.

from __future__ import annotations

import hashlib
from pathlib import Path

from kompend.core.store import open_store, vault_db_path
from kompend.core.store.schema import DOWNLOADED_TABLE
from kompend.logging import get_logger

logger = get_logger(__name__)


class DuplicateDetector:
    """Tracks downloaded URLs and content hashes to detect duplicates.

    Backed by the ``downloaded`` table in the vault store. Each row maps a URL
    to a SHA256 hash of its content.
    """

    def __init__(self, db_file: Path) -> None:
        """Initialize the detector against the vault store.

        Args:
            db_file: The vault artifact path this detector historically owned
                (``.kompend/cache/downloaded.json``). Kept so callers need no
                change; the single store is resolved from it.
        """
        self.db_file = db_file
        self._db_path = vault_db_path(db_file)
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
        with open_store(self._db_path) as conn:
            conn.execute(
                f"INSERT INTO {DOWNLOADED_TABLE}(url, sha256) VALUES (?, ?) ON CONFLICT(url) DO UPDATE SET sha256 = excluded.sha256",
                (url, sha256),
            )

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
        """Load the URL → hash map from the vault store."""
        if not self._db_path.exists():
            self._db = {}
            return
        with open_store(self._db_path) as conn:
            rows = conn.execute(f"SELECT url, sha256 FROM {DOWNLOADED_TABLE}").fetchall()
        self._db = {row["url"]: row["sha256"] for row in rows}
