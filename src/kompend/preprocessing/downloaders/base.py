"""Abstract base for KOMPEND downloaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseDownloader(ABC):
    """Downloader for URLs (PDF, DOCX, websites)."""

    @abstractmethod
    def download(self, url: str, target_dir: Path) -> Path | None:
        """Download URL to target_dir. Returns saved file path or None."""
        ...
