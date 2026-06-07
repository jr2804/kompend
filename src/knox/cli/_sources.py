"""Source file resolution for KNOX CLI.

Handles glob expansion, directory scanning, file copying into the sources
directory, and URL downloading — shared by both `parse` and `compile`.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from knox.preprocessing.downloaders.http_downloader import HTTPDownloader

logger = logging.getLogger(__name__)

# Extensions we know how to handle (lower-case, with dot).
SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf", ".docx", ".doc", ".epub", ".md", ".txt", ".html", ".pptx", ".xlsx",
}


def resolve_files(
    *patterns: str,
    recursive: bool = False,
    ignore_patterns: list[str] | None = None,
    respect_gitignore: bool = True,
) -> list[Path]:
    """Expand a mix of file paths, glob patterns, and directory paths into concrete files.

    Args:
        patterns: File paths, glob patterns, or directory paths.
        recursive: If True, descend into subdirectories.
        ignore_patterns: Glob patterns for files to exclude.
        respect_gitignore: If True, read ``.gitignore`` from source dir.

    Returns:
        Deduplicated, sorted list of existing file paths with supported extensions.
    """
    seen: set[Path] = set()
    results: list[Path] = []

    for raw_pattern in patterns:
        candidate_path = Path(raw_pattern).expanduser()

        # Try as a glob pattern if it contains wildcard characters
        if any(c in raw_pattern for c in ("*", "?", "[")):
            parent_dir = candidate_path.parent if candidate_path.parent != Path(".") else Path.cwd()
            stem = candidate_path.name
            glob_func = parent_dir.rglob if recursive else parent_dir.glob
            for match in sorted(glob_func(stem)):
                if match.is_file() and _is_supported(match):
                    _add(match, seen, results)
            continue

        # Existing file
        if candidate_path.is_file():
            if _is_supported(candidate_path):
                _add(candidate_path, seen, results)
            continue

        # Existing directory — scan it
        if candidate_path.is_dir():
            _scan_dir(candidate_path, recursive, seen, results)
            continue

        # Neither file nor dir nor glob — skip silently.
        # (Could be a typo or an unsupported file that was passed explicitly.)

    # Apply ignore patterns
    if ignore_patterns or respect_gitignore:
        from knox.cli._ignore import filter_files  # noqa: PLC0415

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
) -> list[Path]:
    """Copy files that are not already inside *sources_dir* into it.

    Files already residing under *sources_dir* are returned as-is.
    Returns a new list with paths relative to *sources_dir*.
    """
    sources_dir = sources_dir.resolve()
    sources_dir.mkdir(parents=True, exist_ok=True)

    result: list[Path] = []
    for input_file in input_files:
        resolved_input = input_file.resolve()
        try:
            resolved_input.relative_to(sources_dir)
            # Already inside sources_dir
            result.append(resolved_input)
        except ValueError:
            # Not inside sources_dir — copy it in
            target_file = sources_dir / input_file.name
            target_file = _unique_path(target_file)
            shutil.copy2(resolved_input, target_file)
            logger.info("Copied %s -> %s", input_file, target_file)
            result.append(target_file)

    return result


def download_urls(
    urls: list[str],
    sources_dir: Path,
) -> list[Path]:
    """Download URLs into *sources_dir*, return paths of successful downloads."""
    if not urls:
        return []
    sources_dir = sources_dir.resolve()
    sources_dir.mkdir(parents=True, exist_ok=True)
    downloader = HTTPDownloader()
    results: list[Path] = []
    for url in urls:
        downloaded_file = downloader.download(url, sources_dir)
        if downloaded_file:
            results.append(downloaded_file)
    return results


# ── Internal helpers ────────────────────────────────────────────────────


def _is_supported(candidate_path: Path) -> bool:
    return candidate_path.suffix.lower() in SUPPORTED_EXTENSIONS


def _add(candidate_path: Path, seen: set[Path], results: list[Path]) -> None:
    resolved = candidate_path.resolve()
    if resolved not in seen:
        seen.add(resolved)
        results.append(candidate_path)


def _scan_dir(
    directory: Path,
    recursive: bool,
    seen: set[Path],
    results: list[Path],
) -> None:
    glob_func = directory.rglob if recursive else directory.glob
    for ext in sorted(SUPPORTED_EXTENSIONS):
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
