"""KNOX downloaders for online sources."""

from __future__ import annotations

from knox.preprocessing.downloaders.base import BaseDownloader
from knox.preprocessing.downloaders.crawler import WebCrawler
from knox.preprocessing.downloaders.duplicate_detector import DuplicateDetector
from knox.preprocessing.downloaders.http_downloader import HTTPDownloader

__all__ = [
    "BaseDownloader",
    "DuplicateDetector",
    "HTTPDownloader",
    "WebCrawler",
]
