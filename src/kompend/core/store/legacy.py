"""Detect legacy flat-file ``.kompend/`` artifacts superseded by ``kompend.db``.

This is **detection only, not migration**. Policy B3 is a clean cut: the store
never reads these files. The helper exists so ``kompend doctor`` (Phase 2) can tell
a user which old artifacts are now dead weight and safe to delete after a
``compile --force``.

Deliberately excluded from the candidate set — they are *not* superseded by the
single store and must not be reported:

* ``kompend.db`` itself,
* ``vectors/`` — the separate, regenerable embedding branch,
* ``memory.db`` — a zvec collection directory (rebuildable from ``memory_log``),
* ``config.toml`` / ``rumdl.toml`` — user-editable TOML, and vault markers,
* ``sources/`` / ``wiki/`` / ``attachments/`` — user content.
"""

from __future__ import annotations

from pathlib import Path

from kompend.core.paths import VaultPaths


def obsolete_artifacts(base_dir: Path) -> list[Path]:
    """Return legacy artifacts present on disk that ``kompend.db`` supersedes.

    Each result is an existing path (file or directory). An empty list means the
    vault is already consolidated.
    """
    vault = VaultPaths(base_dir)
    candidates = (
        vault.vectors_manifest_file,
        vault.compile_state_file,
        vault.wiki_cache_file,
        vault.compiled_wiki_file,
        vault.memory_store_file,
        vault.memory_log_file,
        vault.references_file,
        vault.assets_file,
        vault.entity_types_file,
        vault.attachments_manifest_file,
        vault.lint_history_file,
        vault.runs_file,
        vault.organize_report_file,
        vault.verification_file,
        vault.private_file,
        vault.provenance_dir,
        vault.dedup_dir,
        vault.cache_dir,
    )
    return [path for path in candidates if path.exists()]
