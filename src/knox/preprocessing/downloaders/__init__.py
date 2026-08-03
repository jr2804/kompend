"""KNOX downloaders for online sources."""

from __future__ import annotations

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "base": ["BaseDownloader"],
        "crawler": ["WebCrawler"],
        "duplicate_detector": ["DuplicateDetector"],
        "http_downloader": ["HTTPDownloader"],
        "url_resolver": [
            "ArxivHtmlResolver",
            "ArxivPdfResolver",
            "ArxivSourceResolver",
            "BaseUrlResolver",
            "ResolvedUrl",
            "UrlResolverChain",
            "WikipediaPdfResolver",
        ],
    },
)
