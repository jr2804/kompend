"""URL resolver chain — rewrites known content URLs to better download endpoints.

Wikipedia articles, arXiv abstracts, and similar sites provide structured
downloads (PDF, clean HTML) that produce far better results than scraping
the human-facing HTML page.  This module provides a pluggable resolver
chain that recognises known URL patterns and rewrites them before the
download phase.

Adding a new resolver
---------------------
Subclass :class:`BaseUrlResolver`, implement :meth:`resolve`, and add an
instance to the default list in :class:`UrlResolverChain`.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── Models ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResolvedUrl:
    """Outcome of resolving a URL through a resolver.

    Attributes:
        original_url: The URL as the user or pipeline provided it.
        download_url:  The URL to actually download.
        content_type:  MIME type hint for the download
                       (``None`` = auto-detect from HTTP response).
        source_label:  Human-readable label for provenance
                       (e.g. ``"wikipedia"``, ``"arxiv"``).
        prefer_http:   When ``True``, bypass the HTML-URL heuristic and
                       always use :class:`HTTPDownloader` instead of
                       the browser-based ``WebCrawler``. Set this when the
                       URL looks like an HTML page but the resolver has
                       already arranged a direct binary or raw-HTML download
                       (e.g. ``arxiv.org/html/``, ``arxiv.org/src/``).
    """

    original_url: str
    download_url: str
    content_type: str | None = None
    source_label: str | None = None
    prefer_http: bool = False


# ── Resolver base ───────────────────────────────────────────────────────


class BaseUrlResolver(ABC):
    """Abstract base for a URL resolver.

    Implement :meth:`resolve` to return a :class:`ResolvedUrl` when the
    resolver recognises the URL, or ``None`` to let the next resolver
    in the chain try.
    """

    @abstractmethod
    def resolve(self, url: str) -> ResolvedUrl | None: ...


# ── Built-in resolvers ──────────────────────────────────────────────────


class WikipediaPdfResolver(BaseUrlResolver):
    """Resolve ``*.wikipedia.org/wiki/Page_Title`` → PDF API endpoint.

    Wikipedia's REST API produces a professional-quality PDF with all
    equations (MathML/LaTeX), tables, and images rendered server-side.
    Docling already handles PDFs perfectly, so this avoids the quality
    loss of scraping the HTML page.
    """

    _WIKI_RE = re.compile(
        r"^https?://([a-z-]+)\.wikipedia\.org/wiki/(.+)$",
        re.IGNORECASE,
    )

    def resolve(self, url: str) -> ResolvedUrl | None:
        m = self._WIKI_RE.match(url)
        if not m:
            return None
        lang = m.group(1)
        title = m.group(2)
        # Strip URL fragment (#section) — PDF renders the whole article.
        title = title.split("#")[0]
        download_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/pdf/{title}"
        return ResolvedUrl(
            original_url=url,
            download_url=download_url,
            content_type="application/pdf",
            source_label="wikipedia",
        )


class ArxivPdfResolver(BaseUrlResolver):
    """Resolve ``arxiv.org/abs/XXXX.XXXXX`` → direct PDF download.

    The abstract page is HTML with an embedded PDF viewer.  Downloading
    the PDF directly is faster and avoids parsing HTML.
    """

    _ARXIV_RE = re.compile(
        r"^https?://arxiv\.org/abs/(\d+\.\d+[^/]*?)(?:v\d+)?$",
        re.IGNORECASE,
    )

    def resolve(self, url: str) -> ResolvedUrl | None:
        m = self._ARXIV_RE.match(url)
        if not m:
            return None
        paper_id = m.group(1)
        download_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        return ResolvedUrl(
            original_url=url,
            download_url=download_url,
            content_type="application/pdf",
            source_label="arxiv",
        )


class ArxivHtmlResolver(BaseUrlResolver):
    """Resolve ``arxiv.org/html/<id>[v<n>]`` → raw HTML download.

    The HTML version of arXiv papers is served directly from
    ``arxiv.org/html/`` and is readable without JavaScript.  We force
    ``prefer_http=True`` so the URL is fetched with :class:`HTTPDownloader`
    instead of the browser-based ``WebCrawler``, then routed to
    :class:`WebPageParser` (trafilatura) via its ``.html`` extension.
    """

    _ARXIV_HTML_RE = re.compile(
        r"^https?://arxiv\.org/html/([\w./-]+?)$",
        re.IGNORECASE,
    )

    def resolve(self, url: str) -> ResolvedUrl | None:
        m = self._ARXIV_HTML_RE.match(url)
        if not m:
            return None
        return ResolvedUrl(
            original_url=url,
            download_url=url,
            content_type="text/html",
            source_label="arxiv-html",
            prefer_http=True,
        )


class ArxivSourceResolver(BaseUrlResolver):
    """Resolve ``arxiv.org/src/<id>`` → force HTTP download of the source tarball.

    The ``/src/`` endpoint serves a gzip-compressed tar containing the LaTeX
    source.  The URL has no extension so the HTML heuristic would send it to
    the browser crawler.  Setting ``prefer_http=True`` forces
    :class:`HTTPDownloader` to fetch the binary stream and save it under its
    correct extension (``.tar.gz``), after which the archive-extraction step
    in ``download_urls()`` finds the ``.tex`` file for :class:`PandocParser`.
    """

    _ARXIV_SRC_RE = re.compile(
        r"^https?://arxiv\.org/src/([\w./-]+?)$",
        re.IGNORECASE,
    )

    def resolve(self, url: str) -> ResolvedUrl | None:
        m = self._ARXIV_SRC_RE.match(url)
        if not m:
            return None
        return ResolvedUrl(
            original_url=url,
            download_url=url,
            content_type="application/x-tar",
            source_label="arxiv-src",
            prefer_http=True,
        )


# ── Chain ───────────────────────────────────────────────────────────────


class UrlResolverChain:
    """Ordered chain of URL resolvers.

    The first resolver that returns a :class:`ResolvedUrl` wins.  If no
    resolver matches, the original URL is passed through unchanged.
    """

    def __init__(self, resolvers: list[BaseUrlResolver] | None = None) -> None:
        """Initialise with an optional custom resolver list.

        When *resolvers* is ``None``, the default built-in chain is used:

        1. :class:`WikipediaPdfResolver`
        2. :class:`ArxivPdfResolver`
        3. :class:`ArxivHtmlResolver`
        4. :class:`ArxivSourceResolver`
        5. (passthrough — original URL returned as-is)
        """
        self._resolvers: list[BaseUrlResolver]
        if resolvers is not None:
            self._resolvers = list(resolvers)
        else:
            self._resolvers = [
                WikipediaPdfResolver(),
                ArxivPdfResolver(),
                ArxivHtmlResolver(),
                ArxivSourceResolver(),
            ]

    def resolve(self, url: str) -> ResolvedUrl:
        """Run *url* through every resolver in order.

        Returns the first non-``None`` result, or an identity
        :class:`ResolvedUrl` (original URL unchanged) as fallback.
        """
        for resolver in self._resolvers:
            try:
                result = resolver.resolve(url)
                if result is not None:
                    logger.debug(
                        "URL %s resolved by %s -> %s",
                        url,
                        type(resolver).__name__,
                        result.download_url,
                    )
                    return result
            except Exception:
                logger.exception(
                    "Resolver %s failed on %s",
                    type(resolver).__name__,
                    url,
                )
                # Fall through to the next resolver / passthrough.
        return ResolvedUrl(
            original_url=url,
            download_url=url,
            content_type=None,
            source_label=None,
        )
