"""Web crawler for documentation sites and articles."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import os
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from crawl4ai.browser_adapter import UndetectedAdapter
from crawl4ai.utils import get_chromium_path

from kompend.logging import get_logger
from kompend.preprocessing.downloaders.base import BaseDownloader
from kompend.preprocessing.downloaders.duplicate_detector import DuplicateDetector

logger = get_logger(__name__)


class WebCrawler(BaseDownloader):
    """Crawl documentation sites and articles, save as markdown.

    Stealth mode (playwright-stealth) is always enabled. When *undetected* is
    True (default) and a real Edge/Chrome is installed, Patchright
    (undetected browser) is used instead for stronger Cloudflare /
    bot-protection bypass.

    Args:
        detector: Optional DuplicateDetector for URL dedup.
        undetected: When True (default), use Patchright (undetected browser)
            with a real Edge/Chrome installation. Falls back to stealth mode
            automatically when no real browser is installed. Disable to
            force stealth.
    """

    def __init__(
        self,
        detector: DuplicateDetector | None = None,
        undetected: bool = True,
    ) -> None:
        self._detector = detector

        channel = _detect_real_browser_channel() if undetected else None
        if undetected and channel:
            self._use_undetected = True
            self._channel = channel
            self._available = True  # real browser, no bundled chromium needed
        else:
            self._use_undetected = False
            self._channel = None
            self._available = _chromium_available()
            if undetected:
                logger.warning(
                    "No Edge/Chrome installation found — falling back to "
                    "stealth mode. Install Edge or Chrome for stronger "
                    "bot-protection bypass (undetected mode)."
                )
        if not self._available:
            logger.warning(
                "Playwright Chromium is not installed. "
                "HTML/web URLs will fall back to HTTPDownloader. "
                "Run 'playwright install chromium' to enable the web crawler."
            )

    def download(self, url: str, target_dir: Path) -> Path | None:
        """Synchronous wrapper: crawl URL, extract main content as markdown, save."""
        if not self._available:
            return None  # caller falls back to HTTPDownloader
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        try:
            if loop and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    coro = self._download_async(url, target_dir)
                    return pool.submit(lambda: asyncio.run(coro)).result()
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
        target_file = target_dir / f"{safe_domain}.md"

        # Check duplicate detector: skip crawl if URL already known
        if self._detector and self._detector.is_duplicate(url):
            if target_file.exists():
                logger.info("Skipping already-crawled URL: %s", url)
                return target_file
            logger.debug("URL %s was recorded but file missing, re-crawling", url)

        logger.info(
            "Crawling %s -> %s (%s)",
            url,
            target_file,
            "undetected" if self._use_undetected else "stealth",
        )

        browser_config = BrowserConfig(
            headless=True,
            enable_stealth=True,
            chrome_channel=self._channel or "chromium",
        )
        crawler_kwargs: dict = {"config": browser_config}
        if self._use_undetected:
            crawler_kwargs["crawler_strategy"] = AsyncPlaywrightCrawlerStrategy(
                browser_config=browser_config,
                browser_adapter=UndetectedAdapter(),
            )

        async with AsyncWebCrawler(**crawler_kwargs) as crawler:
            result = await crawler.arun(url)
            if not result.success:
                logger.warning(
                    "Crawl4AI blocked or failed for %s: %s",
                    url,
                    result.error_message or "(no error message)",
                )
                return None
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

            target_file.write_text(content, encoding="utf-8")
            if self._detector:
                self._detector.mark_downloaded(url, content_hash)
            logger.info("Saved crawler result to %s", target_file)
            return target_file


def _chromium_available() -> bool:
    """Return True if the Playwright Chromium binary exists on disk."""
    try:
        path = asyncio.run(get_chromium_path("chromium"))
        return bool(path and Path(path).exists())
    except Exception:
        return False


def _detect_real_browser_channel() -> str | None:
    """Return a Playwright ``channel`` name (``"chrome"`` or ``"msedge"``)
    for a real installed browser, else ``None``.

    Patchright (undetected mode) bypasses bot detection best with a *real*
    browser binary rather than bundled chromium.
    """
    pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    # Windows: Chrome preferred, then Edge.
    for exe in (
        rf"{pf}\Google\Chrome\Application\chrome.exe",
        rf"{pf86}\Google\Chrome\Application\chrome.exe",
        rf"{pf86}\Microsoft\Edge\Application\msedge.exe",
        rf"{pf}\Microsoft\Edge\Application\msedge.exe",
    ):
        if Path(exe).exists():
            return "chrome" if "Chrome" in exe else "msedge"
    # macOS
    for channel, app in (
        ("chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        ("msedge", "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
    ):
        if Path(app).exists():
            return channel
    # Linux: on PATH
    for channel, cmd in (("chrome", "google-chrome"), ("msedge", "microsoft-edge")):
        if shutil.which(cmd):
            return channel
    return None
