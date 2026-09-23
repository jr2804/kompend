"""Locate a vault's ``.knox/knox.db`` from a store's path argument.

Phase 2a transition: domain stores historically received the path of their own
artifact file (``.knox/compile_state.json``, ``.knox/cache`` …). The single
store keys everything by the vault root, so each store resolves the root from
whatever path it was handed. New callers should pass the vault ``base_dir``
directly; both forms land on the same DB.

Pure path arithmetic — no filesystem probing and no legacy-file reads.
"""

from __future__ import annotations

from pathlib import Path

from knox.core.paths import KNOX_DIR, VaultPaths


def vault_db_path(path: Path) -> Path:
    """Resolve the single store file for the vault that contains *path*."""
    return VaultPaths(resolve_base_dir(path)).knox_db_file


def resolve_base_dir(path: Path) -> Path:
    """Return the vault root for *path*.

    * the vault ``base_dir`` itself → returned unchanged;
    * any ``<base>/.knox/…`` artifact path → the ``<base>`` above ``.knox``;
    * an existing file with no ``.knox`` ancestor (e.g. an ad-hoc test path) →
      its parent directory.
    """
    if path.name == KNOX_DIR:
        return path.parent
    for parent in path.parents:
        if parent.name == KNOX_DIR:
            return parent.parent
    return path.parent if path.is_file() else path
