"""Source file resolution for KNOX CLI.

Handles glob expansion, scan_dir scanning, file copying into the sources
scan_dir, and URL downloading — shared by both `parse` and `compile`.
"""

from __future__ import annotations

import bz2
import filecmp
import gzip
import hashlib
import lzma
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import py7zr
from unrar.cffi import rarfile

from knox.cli._ignore import filter_files
from knox.core.paths import VaultPaths
from knox.logging import get_logger
from knox.preprocessing.downloaders.crawler import WebCrawler
from knox.preprocessing.downloaders.duplicate_detector import DuplicateDetector
from knox.preprocessing.downloaders.http_downloader import HTTPDownloader
from knox.preprocessing.downloaders.url_resolver import UrlResolverChain
from knox.preprocessing.parsers._registry import get_all_extensions
from knox.preprocessing.parsers.base import FileExtension

logger = get_logger(__name__)


# URL path suffixes whose content is essentially always HTML / a server-rendered
# page. URLs with these (or no extension) go through the browser-based
# WebCrawler; everything else (.pdf, .docx, …) goes through HTTPDownloader.
_HTML_PATH_SUFFIXES: frozenset[str] = frozenset({"", ".html", ".htm", ".aspx", ".asp", ".php", ".jsp", ".cgi", ".jspx"})

# Path segments that mark a URL as binary regardless of extension (e.g. the
# Wikipedia REST API serves ``/api/rest_v1/page/pdf/<Title>`` with no .pdf
# suffix — without this, the heuristic misroutes it to the browser).
_BINARY_PATH_SEGMENTS: frozenset[str] = frozenset({"/pdf/"})


def _looks_like_html_url(url: str) -> bool:
    """Heuristic: does *url*'s path suggest server-rendered HTML?"""
    path = urlparse(url).path.lower()
    if not path or path.endswith("/"):
        return True
    if any(seg in path for seg in _BINARY_PATH_SEGMENTS):
        return False
    return Path(path).suffix in _HTML_PATH_SUFFIXES


# Priority order for picking a file from an extracted archive.
_ARCHIVE_EXTRACT_PRIORITY: tuple[str, ...] = (
    ".tex",
    ".md",
    ".rst",
    ".txt",
    ".html",
    ".htm",
    ".pdf",
    ".docx",
    ".epub",
)


_EXT_PRIORITY: dict[str, int] = {ext: i for i, ext in enumerate(_ARCHIVE_EXTRACT_PRIORITY)}


def _find_best_archive_member(names: list[str]) -> str | None:
    """Find the member with the highest-priority extension."""
    best_name, best_p = None, len(_EXT_PRIORITY)
    for nm in names:
        p = _EXT_PRIORITY.get(Path(nm).suffix.lower(), len(_EXT_PRIORITY))
        if p < best_p:
            best_p, best_name = p, nm
    return best_name


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
    with rarfile.RarFile(path) as rf:
        best = _find_best_archive_member(rf.namelist())
        if best is None:
            return None
        dest = dest_dir / Path(best).name
        dest.write_bytes(rf.read(best))
        return dest


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


def _extract_single_file(path: Path, dest_dir: Path, decompress: object) -> Path | None:
    """Decompress a single-file archive (gz, bz2, xz)."""
    inner_name = Path(path.stem).name
    dest = dest_dir / inner_name
    dest.write_bytes(decompress(path.read_bytes()))  # type: ignore[operator]
    return dest


def _extract_archive(archive_path: Path, dest_dir: Path) -> Path | None:
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
    name_lower = archive_path.name.lower()
    suffix = archive_path.suffix.lower()
    _TAR_ENDINGS = (".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar.lzma", ".tar")
    is_tar = suffix in {".tar", ".tgz"} or any(name_lower.endswith(e) for e in _TAR_ENDINGS)
    is_gz = suffix == ".gz" and not is_tar
    is_bz2 = suffix == ".bz2" and not is_tar
    is_xz = suffix in (".xz", ".lzma") and not is_tar

    try:
        if is_tar:
            dest = _extract_tar(archive_path, dest_dir)
        elif suffix == ".zip":
            dest = _extract_zip(archive_path, dest_dir)
        elif suffix == ".7z":
            dest = _extract_7z(archive_path, dest_dir)
        elif suffix == ".rar":
            dest = _extract_rar(archive_path, dest_dir)
        elif is_gz:
            dest = _extract_single_file(archive_path, dest_dir, gzip.decompress)
        elif is_bz2:
            dest = _extract_single_file(archive_path, dest_dir, bz2.decompress)
        elif is_xz:
            dest = _extract_single_file(archive_path, dest_dir, lzma.decompress)
        else:
            return None

        if dest is None:
            return None

        # Single-file decompression preserves the stem extension;
        # skip if the result isn't a parseable format.
        single_file = is_gz or is_bz2 or is_xz
        if single_file and _EXT_PRIORITY.get(dest.suffix.lower(), len(_EXT_PRIORITY)) >= len(_EXT_PRIORITY):
            dest.unlink(missing_ok=True)
            return None

        archive_path.unlink()
        logger.info("Extracted %s from archive -> %s", dest.name, archive_path.name)
        return dest
    except Exception as exc:
        logger.warning("Could not extract archive %s: %s", archive_path.name, exc)
        return None


# ── Extension helpers ───────────────────────────────────────────────────


def get_supported_extensions() -> set[str]:
    """Return the set of supported file extensions.

    Merges the canonical FileExtension enum with any extensions
    registered by external parsers.
    """
    extensions: set[str] = set(FileExtension)
    try:
        extensions.update(get_all_extensions())
    except Exception as exc:
        logger.debug("Could not load external parser extensions: %s", exc, exc_info=True)
    return extensions


def resolve_files(
    *patterns: str,
    recursive: bool = False,
    ignore_patterns: list[str] | None = None,
    respect_gitignore: bool = True,
    sources_dir: Path | None = None,
) -> list[Path]:
    """Expand a mix of file paths, glob patterns, and scan_dir paths into concrete files.

    Args:
        patterns: File paths, glob patterns, or scan_dir paths.
        recursive: If True, descend into subdirectories.
        ignore_patterns: Glob patterns for files to exclude.
        respect_gitignore: If True, read ``.gitignore`` from source dir.
        sources_dir: When set, bare filenames not found at CWD are searched
            recursively under this directory.

    Returns:
        Deduplicated, sorted list of existing file paths with supported extensions.
    """
    seen: set[Path] = set()
    results: list[Path] = []

    for raw_pattern in patterns:
        candidate_file = Path(raw_pattern).expanduser()

        # Try as a glob pattern if it contains wildcard characters
        if any(c in raw_pattern for c in ("*", "?", "[")):
            parent_dir = candidate_file.parent if candidate_file.parent != Path(".") else Path.cwd()
            stem = candidate_file.name
            glob_func = parent_dir.rglob if recursive else parent_dir.glob
            for match in sorted(glob_func(stem)):
                if match.is_file() and _is_supported(match):
                    _add(match, seen, results)
            continue

        # Existing file (absolute, relative, or CWD-relative)
        if candidate_file.is_file():
            if _is_supported(candidate_file):
                _add(candidate_file, seen, results)
            continue

        # Existing scan_dir — scan it
        if candidate_file.is_dir():
            _scan_dir(candidate_file, recursive, seen, results)
            continue

        # Bare filename not found at CWD — search sources_dir recursively
        if sources_dir is not None and "/" not in raw_pattern and "\\" not in raw_pattern:
            found = False
            for match in sorted(sources_dir.rglob(candidate_file.name)):
                if match.is_file() and _is_supported(match):
                    _add(match, seen, results)
                    found = True
            if found:
                continue

        # Neither file nor dir nor glob — skip silently.
        # (Could be a typo or an unsupported file that was passed explicitly.)

    if ignore_patterns or respect_gitignore:
        results = filter_files(
            results,
            ignore_patterns=ignore_patterns,
            respect_gitignore=respect_gitignore,
            base_dir=None,  # auto-detect from first resolved file's common parent
        )

    return results


def ensure_in_sources(
    input_files: list[Path],
    sources_dir: Path,
    *,
    in_place: bool = False,
) -> list[Path]:
    """Copy files that are not already inside *sources_dir* into it.

    Files already residing under *sources_dir* are returned as-is.
    If a file with the same name and identical content already exists in
    *sources_dir*, the copy is skipped and the existing file is returned.

    When *in_place* is ``True``, files outside *sources_dir* are **not**
    copied — their original paths are returned directly. This is useful for
    large source trees where copying would be wasteful or dangerous.

    Returns a new list of resolved paths.
    """
    sources_dir = sources_dir.resolve()
    if not in_place:
        sources_dir.mkdir(parents=True, exist_ok=True)

    result: list[Path] = []
    for input_file in input_files:
        resolved_input = input_file.resolve()
        try:
            resolved_input.relative_to(sources_dir)
            # Already inside sources_dir
            result.append(resolved_input)
        except ValueError:
            if in_place:
                logger.debug("In-place mode: %s kept at original location", input_file)
                result.append(resolved_input)
                continue
            # Not inside sources_dir — copy it in
            target_file = sources_dir / input_file.name
            if target_file.exists():
                if filecmp.cmp(target_file, resolved_input, shallow=False):
                    logger.debug("Skipping %s — identical file exists in sources", input_file)
                    result.append(target_file)
                    continue
                target_file = _unique_path(target_file)
            shutil.copy2(resolved_input, target_file)
            logger.debug("Copied %s -> %s", input_file, target_file)
            result.append(target_file)

    return result


def download_urls(
    urls: list[str],
    sources_dir: Path,
    *,
    base_dir: Path | None = None,
    undetected: bool = True,
) -> list[Path]:
    """Download URLs into *sources_dir*, return paths of successful downloads.

    Uses a DuplicateDetector to skip URLs that have already been downloaded.
    The detector database is stored in ``base_dir/.knox/downloaded.json``.

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
        base_dir: Vault root used to locate ``.knox/downloaded.json``.
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
    detector = DuplicateDetector(VaultPaths(resolved_base_dir).downloaded_path)
    resolver_chain = UrlResolverChain()
    http_downloader = HTTPDownloader(detector=detector)
    web_crawler = WebCrawler(detector=detector, undetected=undetected)
    results: list[Path] = []
    for url in urls:
        # Resolve: Wikipedia → PDF API, arXiv → direct PDF, etc.
        resolved = resolver_chain.resolve(url)

        # Dedup at the *original* URL level so the same Wikipedia
        # article isn't re-downloaded on every pipeline run.
        if detector.is_duplicate(resolved.original_url):
            logger.debug("Skipping already-downloaded URL: %s", resolved.original_url)
            continue

        use_crawler = not resolved.prefer_http and _looks_like_html_url(resolved.download_url)
        downloader = web_crawler if use_crawler else http_downloader
        downloaded_file = downloader.download(resolved.download_url, sources_dir)
        # If web crawler is unavailable/failed, fall back to HTTPDownloader.
        if downloaded_file is None and use_crawler:
            downloaded_file = http_downloader.download(resolved.download_url, sources_dir)
        if downloaded_file:
            # Expand archives (.tar.gz, .gz, .zip) to find the first
            # parseable document inside them.
            downloaded_file = _extract_archive(downloaded_file, sources_dir) or downloaded_file
            # Track the original URL for future dedup.
            content = downloaded_file.read_bytes()
            content_hash = hashlib.sha256(content).hexdigest()
            detector.mark_downloaded(resolved.original_url, content_hash)
            if resolved.original_url != resolved.download_url:
                logger.info(
                    "Resolved %s → %s → %s",
                    resolved.original_url,
                    resolved.source_label or "?",
                    downloaded_file.name,
                )
            results.append(downloaded_file)
    return results


# ── Internal helpers ────────────────────────────────────────────────────


def _is_supported(candidate_file: Path) -> bool:
    return candidate_file.suffix.lower() in get_supported_extensions()


def _add(candidate_file: Path, seen: set[Path], results: list[Path]) -> None:
    resolved = candidate_file.resolve()
    if resolved not in seen:
        seen.add(resolved)
        results.append(candidate_file)


def _scan_dir(
    scan_dir: Path,
    recursive: bool,
    seen: set[Path],
    results: list[Path],
) -> None:
    glob_func = scan_dir.rglob if recursive else scan_dir.glob
    for ext in sorted(get_supported_extensions()):
        for match in sorted(glob_func(f"*{ext}")):
            if match.is_file():
                _add(match, seen, results)


def _unique_path(target_file: Path) -> Path:
    """If *target_file* exists, append _1, _2, … until unique."""
    if not target_file.exists():
        return target_file
    stem = target_file.stem
    suffix = target_file.suffix
    parent_dir = target_file.parent
    counter = 1
    while True:
        candidate = parent_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
