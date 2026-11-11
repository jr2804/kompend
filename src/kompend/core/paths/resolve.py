"""Derivations from artifact names to paths, slugs and vault-relative labels.

These helpers sit between the name authority (:mod:`kompend.core.paths.names`)
and their callers: they never decide *what* an artifact is called, only how a
given source path maps onto that vocabulary. The package root re-exports them,
so callers import from :mod:`kompend.core.paths`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from kompend.core.paths.names import PROTECTED_SOURCE_SUBDIRS

#: Filename extensions that ``safe_stem`` is allowed to strip. Anything
#: outside this set is preserved verbatim — protects ITU-T filenames like
#: ``T-REC-P.10-201906-I!Amd1!PDF-E.pdf`` where ``.stem`` would mistakenly
#: chop off ``T-REC-P`` because Python treats ``.10-201906-…`` as the
#: extension when ``.pdf`` is absent.
_SAFE_STEM_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".pdf",
        ".html",
        ".htm",
        ".md",
        ".markdown",
        ".txt",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".rst",
        ".tex",
        ".ltx",
        ".eml",
        ".zip",
        ".tar",
        ".gz",
        ".7z",
        ".rar",
        ".docx",
        ".pptx",
        ".xlsx",
        ".doc",
        ".ppt",
        ".xls",
        ".epub",
    }
)

_SLUG_INVALID_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9_-]+")
_SLUG_UNDERSCORE_RE: Final[re.Pattern[str]] = re.compile(r"_+")


def attachment_slugs_for_source(source_rel: Path | str) -> set[str]:
    """All slug forms that could refer to a source file.

    Attachments may be registered with either the full relative path
    (``attachment_slug("papers/thesis.pdf") == "papers_thesis"``) or the
    bare filename (``attachment_slug("thesis.pdf") == "thesis"``) — the
    ``kompend attachments add`` command passes the user's string verbatim.
    Orphan detection and auto-adoption must match BOTH forms: matching one
    form only flags every attachment of a source under a subdirectory as
    orphaned, and ``lint --fix`` then deletes live attachment files.
    """
    rel = Path(source_rel)
    return {attachment_slug(rel), attachment_slug(rel.name)}


def attachment_slug(source_rel: Path | str) -> str:
    """Derive a deterministic flat directory name from a source's relative path.

    The slug is the relative path (without extension) with:
    - slashes replaced by ``_``
    - lowercased
    - non-alphanumeric (except ``_`` and ``-``) collapsed to ``_``
    - leading/trailing underscores stripped

    Examples::

        >>> attachment_slug("report.pdf")
        'report'
        >>> attachment_slug("papers/thesis.pdf")
        'papers_thesis'
        >>> attachment_slug("_captured/quick-note.md")
        'captured_quick-note'
        >>> attachment_slug("reports/Q1/report.pdf")
        'reports_q1_report'
    """
    p = Path(source_rel)
    # Use stem (no extension), join parts with underscore
    parts = list(p.parent.parts) + ([p.stem] if p.stem else [])
    raw = "_".join(parts).lower()
    # Collapse non-alphanumeric to underscore
    raw = _SLUG_INVALID_RE.sub("_", raw)
    # Collapse multiple underscores
    raw = _SLUG_UNDERSCORE_RE.sub("_", raw)
    return raw.strip("_")


def is_protected_source(path: Path, sources_dir: Path) -> bool:
    """True if *path* lives under a protected tool-managed subdir of *sources_dir*.

    Files in ``sources/_captured/``, ``sources/_learned/`` and
    ``sources/_distilled/`` are inputs produced by ``capture``/``learn``/
    ``distill``; ``sources/_archive/`` holds notes retired by
    ``consolidate``/``distill``. ``organize`` must never flatten or relocate
    any of them — they are left in place. Compile still processes the
    capture-lifecycle subdirs, but not ``_archive`` (excluded by
    ``iter_sources`` because those notes were retired, not removed).
    """
    try:
        rel = path.resolve().relative_to(sources_dir.resolve())
    except ValueError:
        return False
    return any(part in PROTECTED_SOURCE_SUBDIRS for part in rel.parts[:-1])


def rel_to_vault(path: Path | str, base_dir: Path) -> str:
    """Convert an absolute path to a vault-relative POSIX string.

    This is the canonical relativization helper for every persistence
    boundary that writes a source path into a ``.kompend/`` artifact
    (references ``citing_source``, assets ``source``, generation
    checkpoints, search index, wiki cache). Keeping paths relative to
    the vault ``base_dir`` is what makes a vault relocatable — see
    Antipattern 6 in ``src/kompend/AGENTS.md``.

    - Accepts a :class:`~pathlib.Path` or a string.
    - Returns a POSIX-normalized relative path (forward slashes) when
      *path* is under *base_dir*.
    - Returns the path as-is (POSIX-normalized) when it is not under
      *base_dir* — so out-of-vault sources degrade gracefully instead of
      crashing. Callers that must guarantee a vault-relative result should
      check the result with :meth:`Path.is_absolute`.
    - Empty input returns ``""``.
    """
    if not path:
        return ""
    p = Path(path)
    try:
        return p.relative_to(base_dir).as_posix()
    except ValueError:
        return p.as_posix()


def safe_stem(path: Path | str) -> str:
    """Return the basename of *path* with a recognized extension stripped.

    Unlike :attr:`pathlib.Path.stem` — which treats the last ``.xxx`` as the
    extension regardless of whether it's a real one — this only strips
    extensions from :data:`_SAFE_STEM_EXTENSIONS`. Multi-dot ITU filenames
    survive intact::

        Path("T-REC-P.10-201906-I!Amd1!PDF-E.pdf").stem
            → "T-REC-P.10-201906-I!Amd1!PDF-E"      (correct for this one)
        safe_stem("T-REC-P.10-201906-I!Amd1!PDF-E.pdf")
            → "T-REC-P.10-201906-I!Amd1!PDF-E"      (correct)
        Path("T-REC-P.10-201906-I!Amd1!PDF-E").stem
            → "T-REC-P"                            (WRONG — chops parent slug)
        safe_stem("T-REC-P.10-201906-I!Amd1!PDF-E")
            → "T-REC-P.10-201906-I!Amd1!PDF-E"      (preserved verbatim)

    Single source of truth for every provenance / label / slug extraction
    (memory #44). Replaces ad-hoc ``Path(...).stem`` calls — those silently
    corrupted ``source_files`` and ``sources[].id`` on the p863 wiki.
    """
    if not path:
        return ""
    name = Path(str(path)).name  # basename only — drops the directory
    if not name:
        return ""
    # Walk from the end looking for a SAFE suffix to strip.
    for ext in _SAFE_STEM_EXTENSIONS:
        if name.lower().endswith(ext) and len(name) > len(ext):
            return name[: -len(ext)]
    return name
