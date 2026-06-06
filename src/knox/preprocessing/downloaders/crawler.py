"""Web crawler for documentation sites and articles."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from urllib.parse import urlparse

from knox.preprocessing.downloaders.base import BaseDownloader

logger = logging.getLogger(__name__)

try:
    from crawl4ai import AsyncWebCrawler

    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False


class WebCrawler(BaseDownloader):
    """Crawl documentation sites and articles, save as markdown."""

    def __init__(self) -> None:
        if not CRAWL4AI_AVAILABLE:
            logger.warning("crawl4ai not installed. Web crawling disabled.")

    def download(self, url: str, target_dir: Path) -> Path | None:
        """Synchronous wrapper: crawl URL, extract main content as markdown, save.

        Args:
            url: The URL to crawl.
            target_dir: Directory to save the extracted markdown file.

        Returns:
            Path to the saved markdown file, or None on failure.
        """
        if not CRAWL4AI_AVAILABLE:
            logger.error("crawl4ai required. Install with: uv pip install crawl4ai")
            return None

        try:
            return asyncio.run(self._download_async(url, target_dir))
        except Exception:
            logger.exception("Failed to crawl %s", url)
            return None

    async def _download_async(self, url: str, target_dir: Path) -> Path | None:
        """Async implementation: crawl URL with crawl4ai and save as markdown."""
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.hostname or "unknown"
        safe_domain = re.sub(r"[^a-zA-Z0-9_.-]", "_", domain)
        target_path = target_dir / f"{safe_domain}.md"

        logger.info("Crawling %s -> %s", url, target_path)

        async with AsyncWebCrawler() as crawler:  # type: ignore[possibly-unbound]
            result = await crawler.arun(url)
            content = getattr(result, "markdown", None)
            if not content:
                logger.warning("No markdown content extracted from %s", url)
                return None
            target_path.write_text(content, encoding="utf-8")
            logger.info("Saved crawler result to %s", target_path)
            return target_path

    def download_sync(self, url: str, target_dir: Path) -> Path | None:
        """Alias for download() — synchronous interface."""
        return self.download(url, target_dir)

    def batch_download(self, urls: list[str], target_dir: Path) -> list[Path]:
        """Download multiple URLs, return paths of successful downloads.

        Args:
            urls: List of URLs to crawl.
            target_dir: Directory to save extracted markdown files.

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