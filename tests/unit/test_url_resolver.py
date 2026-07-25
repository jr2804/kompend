"""Unit tests for the URL resolver chain."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import py7zr

from knox.cli._sources import extract_archive
from knox.preprocessing.downloaders.url_resolver import (
    ArxivHtmlResolver,
    ArxivPdfResolver,
    ArxivSourceResolver,
    ResolvedUrl,
    UrlResolverChain,
    WikipediaPdfResolver,
)


class TestWikipediaPdfResolver:
    resolver = WikipediaPdfResolver()

    def test_resolves_wikipedia_url(self) -> None:
        r = self.resolver.resolve("https://en.wikipedia.org/wiki/Speech_recognition")
        assert r is not None
        assert r.download_url == ("https://en.wikipedia.org/api/rest_v1/page/pdf/Speech_recognition")
        assert r.content_type == "application/pdf"
        assert r.source_label == "wikipedia"
        assert r.original_url == ("https://en.wikipedia.org/wiki/Speech_recognition")

    def test_resolves_with_fragment(self) -> None:
        r = self.resolver.resolve("https://en.wikipedia.org/wiki/Speech_recognition#History")
        assert r is not None
        assert "#History" not in r.download_url

    def test_resolves_language_variant(self) -> None:
        r = self.resolver.resolve("https://de.wikipedia.org/wiki/Spracherkennung")
        assert r is not None
        assert "de.wikipedia.org" in r.download_url
        assert "Spracherkennung" in r.download_url

    def test_resolves_simple_wikipedia(self) -> None:
        r = self.resolver.resolve("https://simple.wikipedia.org/wiki/Earth")
        assert r is not None
        assert "simple.wikipedia.org" in r.download_url
        assert "Earth" in r.download_url

    def test_rejects_non_wikipedia(self) -> None:
        assert self.resolver.resolve("https://example.com/page") is None
        assert self.resolver.resolve("https://wikipedia.org/wiki/Foo") is None


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


class TestArxivHtmlResolver:
    def test_resolves_html_url(self) -> None:
        r = ArxivHtmlResolver().resolve("https://arxiv.org/html/2405.17364v1")
        assert r is not None
        assert r.prefer_http is True
        assert r.content_type == "text/html"
        assert r.source_label == "arxiv-html"

    def test_rejects_abs_url(self) -> None:
        assert ArxivHtmlResolver().resolve("https://arxiv.org/abs/2405.17364") is None

    def test_in_default_chain(self) -> None:
        chain = UrlResolverChain()
        r = chain.resolve("https://arxiv.org/html/2405.17364v1")
        assert r is not None
        assert r.prefer_http is True


class TestArxivSourceResolver:
    def test_resolves_src_url(self) -> None:
        r = ArxivSourceResolver().resolve("https://arxiv.org/src/2405.17364")
        assert r is not None
        assert r.prefer_http is True
        assert r.source_label == "arxiv-src"

    def test_rejects_abs_url(self) -> None:
        assert ArxivSourceResolver().resolve("https://arxiv.org/abs/2405.17364") is None

    def test_in_default_chain(self) -> None:
        chain = UrlResolverChain()
        r = chain.resolve("https://arxiv.org/src/2405.17364")
        assert r is not None
        assert r.prefer_http is True


class TestExtractArchive:
    """Unit tests for _extract_archive — no network needed."""

    def test_extracts_tex_from_targz(self, tmp_path: Path) -> None:

        # Build an in-memory tar.gz with a .tex member.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            tex_content = b"\\documentclass{article}\begin{document}Hello.\\end{document}"
            info = tarfile.TarInfo(name="ms.tex")
            info.size = len(tex_content)
            tf.addfile(info, io.BytesIO(tex_content))
        archive = tmp_path / "paper.tar.gz"
        archive.write_bytes(buf.getvalue())

        extracted = extract_archive(archive, tmp_path)
        assert extracted is not None
        assert extracted.suffix == ".tex"
        assert extracted.read_bytes() == tex_content
        assert not archive.exists()  # archive cleaned up

    def test_prefers_tex_over_txt(self, tmp_path: Path) -> None:

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for name, data in [("readme.txt", b"readme"), ("paper.tex", b"\tex")]:
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        archive = tmp_path / "src.tar.gz"
        archive.write_bytes(buf.getvalue())

        extracted = extract_archive(archive, tmp_path)
        assert extracted is not None
        assert extracted.suffix == ".tex"

    def test_returns_none_for_non_archive(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.7")
        assert extract_archive(pdf, tmp_path) is None
        assert pdf.exists()  # untouched


class TestExtractArchive7z:
    """7z support via py7zr."""

    def test_extracts_tex_from_7z(self, tmp_path: Path) -> None:
        tex_content = b"\\documentclass{article}\begin{document}Hello 7z.\\end{document}"
        archive = tmp_path / "paper.7z"
        with py7zr.SevenZipFile(archive, mode="w") as szf:
            szf.writef(io.BytesIO(tex_content), "ms.tex")
        extracted = extract_archive(archive, tmp_path)
        assert extracted is not None
        assert extracted.suffix == ".tex"
        assert extracted.read_bytes() == tex_content
        assert not archive.exists()

    def test_rar_suffix_is_recognised(self, tmp_path: Path) -> None:
        """_extract_archive attempts .rar (bad content -> None, not silent passthrough)."""
        f = tmp_path / "file.rar"
        f.write_bytes(b"not a real rar")
        result = extract_archive(f, tmp_path)
        assert result is None  # bad content, but suffix was recognised (unrar tried)

    def test_returns_none_for_truly_unsupported(self, tmp_path: Path) -> None:
        """Extensions with no extractor return None immediately."""
        for ext in (".cab", ".exe", ".dmg"):
            f = tmp_path / f"file{ext}"
            f.write_bytes(b"binary")
            assert extract_archive(f, tmp_path) is None, f"{ext} should be unsupported"
