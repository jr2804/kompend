"""Web crawler for documentation sites and articles."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import re
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler

from knox.preprocessing.downloaders.base import BaseDownloader
from knox.preprocessing.downloaders.duplicate_detector import DuplicateDetector

logger = logging.getLogger(__name__)


class WebCrawler(BaseDownloader):
    """Crawl documentation sites and articles, save as markdown."""

    def __init__(self, detector: DuplicateDetector | None = None) -> None:
        self._detector = detector

    def download(self, url: str, target_dir: Path) -> Path | None:
        """Synchronous wrapper: crawl URL, extract main content as markdown, save."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        try:
            if loop and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, self._download_async(url, target_dir)).result()  # ty: ignore
            return asyncio.run(self._download_async(url, target_dir))
        except Exception:
            logger.exception("Failed to crawl %s", url)
            return None

    def batch_download(self, urls: list[str], target_dir: Path) -> list[Path]:
        """Download multiple URLs, return paths of successful downloads."""
        results: list[Path] = []
        for url in urls:
            try:
                path = self.download(url, target_dir)
                if path:
                    results.append(path)
            except Exception:
                logger.exception("Failed to download %s", url)
        return results

    async def _download_async(self, url: str, target_dir: Path) -> Path | None:
        """Async implementation: crawl URL with crawl4ai and save as markdown."""
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.hostname or "unknown"
        safe_domain = re.sub(r"[^a-zA-Z0-9_.-]", "_", domain)
        target_path = target_dir / f"{safe_domain}.md"

        # Check duplicate detector: skip crawl if URL already known
        if self._detector and self._detector.is_duplicate(url):
            if target_path.exists():
                logger.info("Skipping already-crawled URL: %s", url)
                return target_path
            logger.debug("URL %s was recorded but file missing, re-crawling", url)

        logger.info("Crawling %s -> %s", url, target_path)

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url)
            content = getattr(result, "markdown", None)
            if not content:
                logger.warning("No markdown content extracted from %s", url)
                return None

            content_bytes = content.encode("utf-8")
            content_hash = hashlib.sha256(content_bytes).hexdigest()

            # Content-based dedup: skip if identical content already exists
            for existing_file in target_dir.iterdir():
                if existing_file.is_file() and existing_file.suffix == ".md" and existing_file.read_bytes() == content_bytes:
                    logger.info(
                        "Skipping %s — identical content already exists as %s",
                        url,
                        existing_file.name,
                    )
                    if self._detector:
                        self._detector.mark_downloaded(url, content_hash)
                    return existing_file

            target_path.write_text(content, encoding="utf-8")
            if self._detector:
                self._detector.mark_downloaded(url, content_hash)
            logger.info("Saved crawler result to %s", target_path)
            return target_path
