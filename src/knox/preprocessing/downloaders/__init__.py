"""KNOX downloaders for online sources."""

from __future__ import annotations

from knox.preprocessing.downloaders.base import BaseDownloader
from knox.preprocessing.downloaders.crawler import WebCrawler
from knox.preprocessing.downloaders.duplicate_detector import DuplicateDetector
from knox.preprocessing.downloaders.http_downloader import HTTPDownloader
from knox.preprocessing.downloaders.url_resolver import (
    ArxivHtmlResolver,
    ArxivPdfResolver,
    ArxivSourceResolver,
    BaseUrlResolver,
    ResolvedUrl,
    UrlResolverChain,
    WikipediaPdfResolver,
)

__all__ = [
    "ArxivHtmlResolver",
    "ArxivPdfResolver",
    "ArxivSourceResolver",
    "BaseDownloader",
    "BaseUrlResolver",
    "DuplicateDetector",
    "HTTPDownloader",
    "ResolvedUrl",
    "UrlResolverChain",
    "WebCrawler",
    "WikipediaPdfResolver",
]
