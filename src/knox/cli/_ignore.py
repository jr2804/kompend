"""Ignore pattern matching for KNOX source scanning.

Supports ``.gitignore``-style glob patterns via ``pathspec``.
Patterns can come from:
- ``.knox/config.yaml`` (``ignore:`` section)
- ``.gitignore`` in the source directory (respect by default)
- CLI ``--ignore`` flags
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compile_patterns(*pattern_lists: list[str] | None) -> list[str]:
    """Combine multiple pattern lists into one deduplicated list."""
    seen: set[str] = set()
    result: list[str] = []
    for patterns in pattern_lists:
        if not patterns:
            continue
        for p in patterns:
            if p and p not in seen:
                seen.add(p)
                result.append(p)
    return result


def matches_any(path: Path, patterns: list[str], base_dir: Path | None = None) -> bool:
    """Check if *path* matches any of the given glob patterns.

    Each pattern is tested against the path's name and, when possible, against
    the path relative to *base_dir*.
    """
    if not patterns:
        return False

    # Compute relative path string (best-effort — may not be under base_dir)
    try:
        rel_path = path.relative_to(base_dir) if base_dir else path
    except ValueError:
        rel_path = path
    rel_str = str(rel_path).replace("\\", "/")
    name = rel_path.name

    for pattern in patterns:
        p = pattern.replace("\\", "/")
        # Try matching against the relative path string
        if _glob_match(rel_str, p):
            return True
        # Try matching against just the filename
        if _glob_match(name, p):
            return True
        # Try matching against the full path (for **/patterns)
        if p.startswith("**/"):
            suffix = p[3:]
            if _glob_match(rel_str, suffix) or rel_str.endswith(f"/{suffix}"):
                return True
            if suffix.endswith("/*"):
                prefix = suffix[:-2]
                if prefix and prefix in rel_str:
                    return True

    return False


def _glob_match(text: str, pattern: str) -> bool:
    """Simple glob matching, used as fallback."""
    import fnmatch  # noqa: PLC0415

    return fnmatch.fnmatch(text, pattern)


def load_gitignore(source_dir: Path) -> list[str]:
    """Load patterns from ``.gitignore`` in *source_dir* (or its parents)."""
    patterns: list[str] = []

    # Walk up from source_dir to find .gitignore
    current = source_dir.resolve()
    for _ in range(10):  # limit depth
        gitignore = current / ".gitignore"
        if gitignore.exists():
            try:
                text = gitignore.read_text(encoding="utf-8", errors="replace")
                for line in text.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
                logger.debug("Loaded %d patterns from %s", len(patterns), gitignore)
            except Exception as exc:
                logger.debug("Failed to read %s: %s", gitignore, exc)
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    return patterns


def filter_files(
    files: list[Path],
    ignore_patterns: list[str] | None = None,
    respect_gitignore: bool = True,
    base_dir: Path | None = None,
) -> list[Path]:
    """Filter out files matching any ignore pattern."""
    # CLI patterns override; config patterns are merged upstream in commands.py
    all_patterns = list(ignore_patterns or [])

    if respect_gitignore and base_dir:
        gitignore_patterns = load_gitignore(base_dir)
        all_patterns.extend(gitignore_patterns)

    if not all_patterns:
        return files

    result: list[Path] = []
    skipped = 0
    for f in files:
        if matches_any(f, all_patterns, base_dir=base_dir):
            skipped += 1
        else:
            result.append(f)

    if skipped:
        logger.info("Skipped %d file(s) matching ignore patterns", skipped)

    return result
