"""Archive extraction and URL downloading for KNOX source resolution.

Extracted from :mod:`knox.cli._sources` — I/O-heavy operations that don't
depend on the file-resolution logic.
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import lzma
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import py7zr
from unrar.cffi import rarfile

from knox.core.paths import VaultPaths
from knox.logging import get_logger
from knox.preprocessing.downloaders.crawler import WebCrawler
from knox.preprocessing.downloaders.duplicate_detector import DuplicateDetector
from knox.preprocessing.downloaders.http_downloader import HTTPDownloader
from knox.preprocessing.downloaders.url_resolver import UrlResolverChain
from knox.preprocessing.parsers.base import FileExtension

# ── URL routing ─────────────────────────────────────────────────────────


# URL path suffixes whose content is essentially always HTML / a server-rendered
# page. URLs with these (or no extension) go through the browser-based
# WebCrawler; everything else (.pdf, .docx, …) goes through HTTPDownloader.
_HTML_PATH_SUFFIXES: frozenset[str] = frozenset({"", ".html", ".htm", ".aspx", ".asp", ".php", ".jsp", ".cgi", ".jspx"})

# Path segments that mark a URL as binary regardless of extension (e.g. the
# Wikipedia REST API serves ``/api/rest_v1/page/pdf/<Title>`` with no .pdf
# suffix — without this, the heuristic misroutes it to the browser).
_BINARY_PATH_SEGMENTS: frozenset[str] = frozenset({"/pdf/"})


# ── Archive extraction ──────────────────────────────────────────────────


# Priority order for picking a file from an extracted archive.
# Lower index = higher priority (TeX > Markdown > plain text > HTML > PDF > DOCX > EPUB).
_ARCHIVE_EXTRACT_PRIORITY: tuple[FileExtension, ...] = (
    FileExtension.TEX,
    FileExtension.MD,
    FileExtension.RST,
    FileExtension.TXT,
    FileExtension.HTML,
    FileExtension.HTM,
    FileExtension.PDF,
    FileExtension.DOCX,
    FileExtension.EPUB,
)


_EXT_PRIORITY: dict[str, int] = {ext.value: i for i, ext in enumerate(_ARCHIVE_EXTRACT_PRIORITY)}

logger = get_logger(__name__)


# ── URL downloading ─────────────────────────────────────────────────────


def download_urls(
    urls: list[str],
    sources_dir: Path,
    *,
    base_dir: Path | None = None,
    undetected: bool = True,
) -> list[Path]:
    """Download URLs into *sources_dir*, return paths of successful downloads.

    Uses a DuplicateDetector to skip URLs that have already been downloaded.
    The detector's URL → hash map lives in the ``downloaded`` table of the
    single ``.knox/knox.db`` store.

    URLs are first passed through :class:`UrlResolverChain` so that
    recognised sites (Wikipedia, arXiv, …) are rewritten to better
    download endpoints (PDF API, direct PDF, …) before fetching.

    Routing:

    - URLs that look like HTML pages (no extension, ``.html``, ``.aspx``,
      ``.php``, …) go through :class:`WebCrawler` (crawl4ai, browser-based,
      markdown extraction, stealth-enabled by default).
    - Everything else (``.pdf``, ``.docx``, …) goes through
      :class:`HTTPDownloader` (urllib, direct bytes).

    Args:
        urls: URLs to download.
        sources_dir: Target directory for downloaded files.
        base_dir: Vault root used to locate the ``.knox/knox.db`` store.
            When ``None``, defaults to ``sources_dir.parent``.
        undetected: When True (default), use Patchright undetected-browser
            mode with a real Edge/Chrome for stronger bot-protection bypass.
            Set via ``downloader.undetected`` config.

    Returns:
        Paths of successfully downloaded files (may be shorter than *urls*).
    """
    if not urls:
        return []
    sources_dir = sources_dir.resolve()
    sources_dir.mkdir(parents=True, exist_ok=True)
    resolved_base_dir = base_dir if base_dir is not None else sources_dir.parent
    detector = DuplicateDetector(VaultPaths(resolved_base_dir).downloaded_file)
    resolver_chain = UrlResolverChain()
    http_downloader = HTTPDownloader(detector=detector)
    web_crawler = WebCrawler(detector=detector, undetected=undetected)
    results: list[Path] = []
    for url in urls:
        dest = _download_single_url(url, resolver_chain, detector, http_downloader, web_crawler, sources_dir)
        if dest:
            results.append(dest)
    return results


def _download_single_url(
    url: str,
    resolver_chain: UrlResolverChain,
    detector: DuplicateDetector,
    http_downloader: HTTPDownloader,
    web_crawler: WebCrawler,
    sources_dir: Path,
) -> Path | None:
    """Download one URL: resolve, fetch, extract, track. Returns destination path or None."""
    resolved = resolver_chain.resolve(url)

    if detector.is_duplicate(resolved.original_url):
        logger.debug("Skipping already-downloaded URL: %s", resolved.original_url)
        return None

    use_crawler = not resolved.prefer_http and looks_like_html_url(resolved.download_url)
    downloader: HTTPDownloader | WebCrawler = web_crawler if use_crawler else http_downloader
    downloaded_file = downloader.download(resolved.download_url, sources_dir)
    if downloaded_file is None and use_crawler:
        downloaded_file = http_downloader.download(resolved.download_url, sources_dir)

    if not downloaded_file:
        return None

    downloaded_file = extract_archive(downloaded_file, sources_dir) or downloaded_file
    content = downloaded_file.read_bytes()
    content_hash = hashlib.sha256(content).hexdigest()
    detector.mark_downloaded(resolved.original_url, content_hash)

    if resolved.original_url != resolved.download_url:
        logger.info(
            "Resolved %s → %s → %s",
            resolved.original_url,
            resolved.source_label or "unknown",
            downloaded_file.name,
        )
    return downloaded_file


def looks_like_html_url(url: str) -> bool:
    """Heuristic: does *url*'s path suggest server-rendered HTML?"""
    path = urlparse(url).path.lower()
    if not path or path.endswith("/"):
        return True
    if any(seg in path for seg in _BINARY_PATH_SEGMENTS):
        return False
    return Path(path).suffix in _HTML_PATH_SUFFIXES


def extract_archive(archive_file: Path, dest_dir: Path) -> Path | None:
    """Expand a compressed archive and return the best inner file.

    Supported formats (stdlib only, no extra dependencies):

    - ``.tar``, ``.tar.gz`` / ``.tgz``, ``.tar.bz2``, ``.tar.xz`` / ``.tar.lzma``
    - ``.gz`` (single-file gzip)
    - ``.bz2`` (single-file bzip2)
    - ``.xz`` / ``.lzma`` (single-file LZMA)
    - ``.zip``

    Supported with ``py7zr`` (optional but listed as a dependency):

    - ``.7z``

    Supported with ``unrar2-cffi`` (listed as a dependency):

    - ``.rar``

    Walks members in name order and returns the first file whose extension
    matches _ARCHIVE_EXTRACT_PRIORITY. The archive is deleted on success.
    Returns None when the path is not a recognised archive.
    """
    name_lower = archive_file.name.lower()
    suffix = archive_file.suffix.lower()
    _TAR_ENDINGS = (".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar.lzma", ".tar")
    is_tar = suffix in {".tar", ".tgz"} or any(name_lower.endswith(e) for e in _TAR_ENDINGS)
    is_gz = suffix == ".gz" and not is_tar
    is_bz2 = suffix == ".bz2" and not is_tar
    is_xz = suffix in (".xz", ".lzma") and not is_tar

    try:
        if is_tar:
            dest = _extract_tar(archive_file, dest_dir)
        elif suffix == ".zip":
            dest = _extract_zip(archive_file, dest_dir)
        elif suffix == ".7z":
            dest = _extract_7z(archive_file, dest_dir)
        elif suffix == ".rar":
            dest = _extract_rar(archive_file, dest_dir)
        elif is_gz:
            dest = _extract_single_file(archive_file, dest_dir, gzip.decompress)
        elif is_bz2:
            dest = _extract_single_file(archive_file, dest_dir, bz2.decompress)
        elif is_xz:
            dest = _extract_single_file(archive_file, dest_dir, lzma.decompress)
        else:
            return None

        if dest is None:
            return None

        dest = _clean_single_file_extract(dest, is_gz or is_bz2 or is_xz)
        if dest is None:
            return None

        archive_file.unlink()
        logger.info("Extracted %s from archive -> %s", dest.name, archive_file.name)
        return dest
    except Exception as exc:
        logger.warning("Could not extract archive %s: %s", archive_file.name, exc)
        return None


def _extract_7z(path: Path, dest_dir: Path) -> Path | None:
    with py7zr.SevenZipFile(path, mode="r") as szf:
        best = _find_best_archive_member(szf.getnames())
        if best is None:
            return None
        szf.extract(targets=[best], path=dest_dir)
        dest = dest_dir / Path(best).name
        if not dest.exists():
            dest = next(dest_dir.rglob(Path(best).name), None)
        return dest


def _extract_zip(path: Path, dest_dir: Path) -> Path | None:
    with zipfile.ZipFile(path) as zf:
        best = _find_best_archive_member(zf.namelist())
        if best is None:
            return None
        dest = dest_dir / Path(best).name
        dest.write_bytes(zf.read(best))
        return dest


def _extract_rar(path: Path, dest_dir: Path) -> Path | None:
    if not rarfile.is_rarfile(path):
        return None
    rf: rarfile.RarFile = rarfile.RarFile(path)
    try:
        best = _find_best_archive_member(rf.namelist())
        if best is None:
            return None
        dest = dest_dir / Path(best).name
        dest.write_bytes(rf.read(best))
        return dest
    finally:
        pass


def _extract_tar(path: Path, dest_dir: Path) -> Path | None:
    with tarfile.open(path, mode="r:*") as tf:
        file_members = [m for m in tf.getmembers() if m.isfile()]
        best = _find_best_archive_member([m.name for m in file_members])
        if best is None:
            return None
        member = next(m for m in file_members if m.name == best)
        fobj = tf.extractfile(member)
        if fobj is None:
            return None
        dest = dest_dir / Path(best).name
        dest.write_bytes(fobj.read())
        return dest


def _find_best_archive_member(names: list[str]) -> str | None:
    """Find the member with the highest-priority extension."""
    best_name, best_p = None, len(_EXT_PRIORITY)
    for nm in names:
        p = _EXT_PRIORITY.get(Path(nm).suffix.lower(), len(_EXT_PRIORITY))
        if p < best_p:
            best_p, best_name = p, nm
    return best_name


def _extract_single_file(path: Path, dest_dir: Path, decompress: Callable[[bytes], bytes]) -> Path | None:
    """Decompress a single-file archive (gz, bz2, xz)."""
    inner_name = Path(path.stem).name
    dest = dest_dir / inner_name
    dest.write_bytes(decompress(path.read_bytes()))
    return dest


def _clean_single_file_extract(dest: Path, is_single_file: bool) -> Path | None:
    """Remove single-file extract result when it's not a parseable format."""
    if is_single_file and _EXT_PRIORITY.get(dest.suffix.lower(), len(_EXT_PRIORITY)) >= len(_EXT_PRIORITY):
        dest.unlink(missing_ok=True)
        return None
    return dest
