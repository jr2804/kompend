"""Workspace checks: the vault's directories, git, disk headroom, the parser.

These answer questions about the machine the run happens on, not about a
model: whether the pipeline has somewhere to read sources from and write pages
to, whether there is room for the next ingest, and whether an ingestion backend
builds at all.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Final

from kompend.diagnostics.base import CheckResult
from kompend.preprocessing.parsers import PARSER_PARSECRAFT, build_parser

#: Free disk space below this is a failure: an ingest of a real corpus needs
#: room for the parse cache and the rendered figures.
MIN_FREE_DISK_MB: Final[int] = 500


def check_sources_dir(sources_dir: Path | None) -> CheckResult:
    """Validate that the sources directory exists and is a directory."""
    name = "sources-directory"
    if sources_dir is None:
        return CheckResult(name, True, "No sources dir configured — skipped")
    if not sources_dir.exists():
        return CheckResult(name, False, f"Sources dir does not exist: {sources_dir}")
    if not sources_dir.is_dir():
        return CheckResult(name, False, f"Sources path is not a directory: {sources_dir}")
    return CheckResult(name, True, f"Sources dir OK: {sources_dir}")


def check_output_dir(output_dir: Path | None) -> CheckResult:
    """Validate that the output directory exists and is writable."""
    name = "output-directory"
    if output_dir is None:
        return CheckResult(name, True, "No output dir configured — skipped")
    if not output_dir.exists():
        return CheckResult(name, False, f"Output dir does not exist: {output_dir}")
    if not os.access(output_dir, os.W_OK):
        return CheckResult(name, False, f"Output dir not writable: {output_dir}")
    return CheckResult(name, True, f"Output dir OK: {output_dir}")


def check_parsers() -> CheckResult:
    """Check that the ingestion parser builds and report how many formats it claims."""
    name = "parsers"
    parser = build_parser().parser
    formats = len(parser.supported_extensions)
    return CheckResult(name, True, f"{PARSER_PARSECRAFT}: {formats} format(s)")


def check_git_repo() -> CheckResult:
    """Check if we are inside a git repository."""
    name = "git-repo"
    # Traverse up from cwd looking for a .git directory — a pure Path check, so
    # no subprocess is spawned and nothing can hang.
    try:
        current_dir = Path.cwd().resolve()
        for parent in [current_dir, *current_dir.parents]:
            if (parent / ".git").is_dir():
                return CheckResult(name, True, "Inside a git repository")
        return CheckResult(name, True, "Not a git repository — OK")
    except OSError as exc:
        return CheckResult(name, True, f"Git check failed ({exc}) — OK")


def check_disk_space(sources_dir: Path | None = None) -> CheckResult:
    """Check available disk space — warn if below :data:`MIN_FREE_DISK_MB`."""
    name = "disk-space"
    check_dir = sources_dir if sources_dir and sources_dir.exists() else Path.cwd()
    try:
        usage = shutil.disk_usage(str(check_dir))
    except OSError as exc:
        return CheckResult(name, True, f"Could not check disk space ({exc}) — skipped")
    free_mb = usage.free / (1024 * 1024)
    if free_mb < MIN_FREE_DISK_MB:
        return CheckResult(name, False, f"Low disk space: {free_mb:.0f} MB free (< {MIN_FREE_DISK_MB} MB)")
    return CheckResult(name, True, f"{free_mb:.0f} MB free")
