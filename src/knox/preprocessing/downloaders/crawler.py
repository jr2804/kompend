"""Web crawler for documentation sites and articles."""

from __future__ import annotations

import logging
from pathlib import Path

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
        if not CRAWL4AI_AVAILABLE:
            logger.error("crawl4ai required. Install with: uv pip install crawl4ai")
            return None

        # Stub: implement actual crawl logic with crawl4ai
        target_path = target_dir / f"{url.replace('https://', '').replace('/', '_')}.md"
        logger.info("Crawling %s -> %s", url, target_path)
        return target_path
