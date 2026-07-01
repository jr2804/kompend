"""Source file resolution for KNOX CLI.

Handles glob expansion, scan_dir scanning, file copying into the sources
scan_dir, and URL downloading — shared by both `parse` and `compile`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from knox.cli._ignore import filter_files
from knox.core.paths import VaultPaths
from knox.logging import get_logger
from knox.preprocessing.downloaders.duplicate_detector import DuplicateDetector
from knox.preprocessing.downloaders.http_downloader import HTTPDownloader
from knox.preprocessing.parsers._registry import get_all_extensions
from knox.preprocessing.parsers.base import FileExtension

logger = get_logger(__name__)


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
                if target_file.read_bytes() == resolved_input.read_bytes():
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
) -> list[Path]:
    """Download URLs into *sources_dir*, return paths of successful downloads.

    Uses a DuplicateDetector to skip URLs that have already been downloaded.
    The detector database is stored in ``base_dir/.knox/downloaded.json``.
    """
    if not urls:
        return []
    sources_dir = sources_dir.resolve()
    sources_dir.mkdir(parents=True, exist_ok=True)
    resolved_base_dir = base_dir if base_dir is not None else sources_dir.parent
    detector = DuplicateDetector(VaultPaths(resolved_base_dir).downloaded_path)
    downloader = HTTPDownloader(detector=detector)
    results: list[Path] = []
    for url in urls:
        downloaded_file = downloader.download(url, sources_dir)
        if downloaded_file:
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
