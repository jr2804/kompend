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

__all__ = ["repo_root", "require_scanned"]


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
