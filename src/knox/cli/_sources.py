"""Source file resolution for KNOX CLI.

Handles glob expansion, scan_dir scanning, and file copying into the sources
scan_dir — shared by both ``parse`` and ``compile``.

Archive extraction and URL downloading live in :mod:`knox.cli._sources_io`.
"""

from __future__ import annotations

import filecmp
import shutil
from pathlib import Path

from knox.cli._ignore import filter_files
from knox.cli._sources_io import (
    download_urls,
    extract_archive,
    looks_like_html_url,
)
from knox.logging import get_logger
from knox.preprocessing.parsers._registry import get_all_extensions
from knox.preprocessing.parsers.base import FileExtension

logger = get_logger(__name__)

__all__ = [
    "download_urls",
    "ensure_in_sources",
    "extract_archive",
    "get_supported_extensions",
    "looks_like_html_url",
    "resolve_files",
]


# ── File resolution ─────────────────────────────────────────────────────


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
        if _resolve_glob(raw_pattern, candidate_file, recursive, seen, results):
            continue
        if _resolve_file(candidate_file, seen, results):
            continue
        if _resolve_dir(candidate_file, recursive, seen, results):
            continue
        if _resolve_by_name(candidate_file, raw_pattern, sources_dir, seen, results):
            continue
        # Neither file nor dir nor glob — skip silently.

    if ignore_patterns or respect_gitignore:
        results = filter_files(
            results,
            ignore_patterns=ignore_patterns,
            respect_gitignore=respect_gitignore,
            base_dir=None,
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


# ── Internal helpers ────────────────────────────────────────────────────


def _resolve_glob(
    raw_pattern: str,
    candidate_file: Path,
    recursive: bool,
    seen: set[Path],
    results: list[Path],
) -> bool:
    """Resolve a glob pattern and add matching files. Returns True if pattern was a glob."""
    if not any(c in raw_pattern for c in ("*", "?", "[")):
        return False
    parent_dir = candidate_file.parent if candidate_file.parent != Path(".") else Path.cwd()
    stem = candidate_file.name
    glob_func = parent_dir.rglob if recursive else parent_dir.glob
    for match in sorted(glob_func(stem)):
        if match.is_file() and _is_supported(match):
            _add(match, seen, results)
    return True


def _resolve_file(candidate_file: Path, seen: set[Path], results: list[Path]) -> bool:
    """Add candidate_file if it exists and is a supported format."""
    if not candidate_file.is_file():
        return False
    if _is_supported(candidate_file):
        _add(candidate_file, seen, results)
    return True


def _resolve_dir(
    candidate_file: Path,
    recursive: bool,
    seen: set[Path],
    results: list[Path],
) -> bool:
    """Scan a directory and add supported files."""
    if not candidate_file.is_dir():
        return False
    _scan_dir(candidate_file, recursive, seen, results)
    return True


def _resolve_by_name(
    candidate_file: Path,
    raw_pattern: str,
    sources_dir: Path | None,
    seen: set[Path],
    results: list[Path],
) -> bool:
    """Search for a bare filename under sources_dir. Returns True if found."""
    if sources_dir is None or "/" in raw_pattern or "\\" in raw_pattern:
        return False
    found = False
    for match in sorted(sources_dir.rglob(candidate_file.name)):
        if match.is_file() and _is_supported(match):
            _add(match, seen, results)
            found = True
    return found


def _is_supported(candidate_file: Path) -> bool:
    return candidate_file.suffix.lower() in get_supported_extensions()


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


def _add(candidate_file: Path, seen: set[Path], results: list[Path]) -> None:
    resolved = candidate_file.resolve()
    if resolved not in seen:
        seen.add(resolved)
        results.append(candidate_file)


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
