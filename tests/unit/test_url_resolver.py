"""Unit tests for the URL resolver chain."""

from __future__ import annotations

import pytest

from knox.preprocessing.downloaders.url_resolver import (
    ArxivPdfResolver,
    ResolvedUrl,
    UrlResolverChain,
    WikipediaPdfResolver,
)


class TestWikipediaPdfResolver:
    resolver = WikipediaPdfResolver()

    def test_resolves_wikipedia_url(self) -> None:
        r = self.resolver.resolve(
            "https://en.wikipedia.org/wiki/Speech_recognition"
        )
        assert r is not None
        assert r.download_url == (
            "https://en.wikipedia.org/api/rest_v1/page/pdf/"
            "Speech_recognition"
        )
        assert r.content_type == "application/pdf"
        assert r.source_label == "wikipedia"
        assert r.original_url == (
            "https://en.wikipedia.org/wiki/Speech_recognition"
        )

    def test_resolves_with_fragment(self) -> None:
        r = self.resolver.resolve(
            "https://en.wikipedia.org/wiki/Speech_recognition#History"
        )
        assert r is not None
        assert "#History" not in r.download_url

    def test_resolves_language_variant(self) -> None:
        r = self.resolver.resolve(
            "https://de.wikipedia.org/wiki/Spracherkennung"
        )
        assert r is not None
        assert "de.wikipedia.org" in r.download_url
        assert "Spracherkennung" in r.download_url

    def test_resolves_simple_wikipedia(self) -> None:
        r = self.resolver.resolve(
            "https://simple.wikipedia.org/wiki/Earth"
        )
        assert r is not None
        assert "simple.wikipedia.org" in r.download_url
        assert "Earth" in r.download_url

    def test_rejects_non_wikipedia(self) -> None:
        assert self.resolver.resolve("https://example.com/page") is None
        assert (
            self.resolver.resolve("https://wikipedia.org/wiki/Foo")
            is None
        )


class TestArxivPdfResolver:
    resolver = ArxivPdfResolver()

    def test_resolves_arxiv_abs(self) -> None:
        r = self.resolver.resolve("https://arxiv.org/abs/2301.12345")
        assert r is not None
        assert r.download_url == "https://arxiv.org/pdf/2301.12345.pdf"
        assert r.content_type == "application/pdf"
        assert r.source_label == "arxiv"

    def test_resolves_with_version(self) -> None:
        r = self.resolver.resolve("https://arxiv.org/abs/2301.12345v2")
        assert r is not None
        assert r.download_url == "https://arxiv.org/pdf/2301.12345.pdf"

    def test_rejects_non_arxiv(self) -> None:
        assert self.resolver.resolve("https://example.com/abs/2301.12345") is None


class TestUrlResolverChain:
    def test_resolves_wikipedia(self) -> None:
        chain = UrlResolverChain()
        r = chain.resolve("https://en.wikipedia.org/wiki/Speech_recognition")
        assert isinstance(r, ResolvedUrl)
        assert r.source_label == "wikipedia"

    def test_resolves_arxiv(self) -> None:
        chain = UrlResolverChain()
        r = chain.resolve("https://arxiv.org/abs/2301.12345")
        assert isinstance(r, ResolvedUrl)
        assert r.source_label == "arxiv"

    def test_passthrough_unknown(self) -> None:
        chain = UrlResolverChain()
        r = chain.resolve("https://example.com/some/page.html")
        assert isinstance(r, ResolvedUrl)
        assert r.original_url == r.download_url
        assert r.source_label is None
        assert r.content_type is None

    def test_custom_chain(self) -> None:
        """A chain with no resolvers should passthrough everything."""
        chain = UrlResolverChain(resolvers=[])
        r = chain.resolve("https://en.wikipedia.org/wiki/Foo")
        assert r.original_url == r.download_url
        assert r.source_label is None
