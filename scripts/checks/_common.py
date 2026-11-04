"""Shared helpers for the ``scripts/checks`` gate scripts.

Both helpers exist so a gate fails loudly instead of passing vacuously.
The checks previously derived the repository root by counting ``parent``
hops from their own location; when they moved into ``scripts/checks/`` the
root silently became ``scripts/``, every scan found zero files, and the
gate reported success.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path


class ScanError(RuntimeError):
    """The scan itself could not run — unreadable or unparseable input.

    Not a finding, and not a pass: the caller turns it into exit code 2. Shared
    so every gate raises the same type instead of each defining its own.
    """


__all__ = ["ScanError", "repo_root", "require_facts", "require_scanned"]


def repo_root() -> Path:
    """Return the repository root, located by its ``pyproject.toml`` marker.

    Walking up to a marker rather than counting hops means moving these
    scripts to another directory cannot relocate the root by accident.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"repository root (pyproject.toml) not found above {__file__}")


def require_scanned(count: int, roots: Iterable[Path]) -> None:
    """Abort when a scan covered no files — an empty scan is never a pass."""
    if count > 0:
        return
    joined = ", ".join(str(root) for root in roots)
    print(
        f"ERROR: nothing to scan under {joined} — refusing to report success",
        file=sys.stderr,
    )
    raise SystemExit(2)


def require_facts(count: int, what: str, remedy: str) -> None:
    """Abort when a rule's *facts* came back empty — no facts, no verdict.

    Distinct from :func:`require_scanned`, and needed for the same reason one
    level deeper: a gate that reads its files but derives its judgement from an
    empty fact set reports the *absence of facts* as the absence of violations.
    A glob that matched nothing, a registry that moved, a config model that
    stopped exporting defaults — none of them raises, all of them disarm the
    rule silently.

    *remedy* must name the fix; an unactionable error is only slightly better
    than a false pass.

    Precondition: call this only where zero is *never* a legitimate answer.
    Zero has three meanings and this helper can only see the count —
    "declares none" (an honest empty list), "not applicable" (a virtual
    workspace root with no ``[project]`` table) and "could not derive it".
    Passing the first two here trades a silent pass for a loud false alarm,
    so a caller whose facts can legitimately be empty must check the shape
    first and abort itself. Every current call site derives from a source
    that cannot honestly be empty (config models, the cache-enum registry,
    the CLI command tree).
    """
    if count > 0:
        return
    print(
        f"ERROR: could not collect any {what} — the scan would report every violation as absent. {remedy}",
        file=sys.stderr,
    )
    raise SystemExit(2)
