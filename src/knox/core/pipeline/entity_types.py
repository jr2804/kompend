"""Per-vault entity-type registry.

Stable per-concept entity kinds, persisted across recompiles so the
folder assignment under ``entities/<type>/<id>.md`` does not flicker
when the model gives a slightly different ``type`` value for the same
concept on different runs.

Backed by the ``entity_types`` table of the single ``.knox/knox.db`` store.
Keys are :func:`knox.compilation.concept_keys.normalize_concept_key`
of the concept name; values are sanitized folder names (see
:func:`knox.compilation.flows.generation._sanitize_entity_type`).

See p863 wiki roundup (2026-09-10) for the motivating bug — the
model returned different ``type`` values for POLQA across runs,
causing its page to land in ``entities/standard/`` one time and
``entities/polqa/`` the next.
"""

# ruff: noqa: S608 — SQL is built from Final[str] table-name constants only.

from __future__ import annotations

from pathlib import Path

from knox.core.store import open_store, vault_db_path
from knox.core.store.schema import ENTITY_TYPES_TABLE
from knox.logging import get_logger

logger = get_logger(__name__)


class EntityTypeRegistry:
    """Store mapping concept keys to sanitized entity kinds.

    Backed by the ``entity_types`` table of ``.knox/knox.db``.

    Parameters
    ----------
    entity_types_file:
        Legacy artifact path (``.knox/entity_types.json``) used to locate the
        vault root, or the vault ``base_dir`` itself.
    base_dir:
        Vault root. When given, *entity_types_file* is ignored.
    """

    def __init__(self, entity_types_file: Path, base_dir: Path | None = None) -> None:
        self._file = entity_types_file
        self._db_path = vault_db_path(base_dir if base_dir is not None else entity_types_file)
        self._data: dict[str, str] = {}
        self.load()

    def load(self) -> None:
        """Read the registry into memory. A missing store is a fresh start."""
        if not self._db_path.exists():
            self._data = {}
            return
        with open_store(self._db_path) as conn:
            rows = conn.execute(f"SELECT concept_key, folder_name FROM {ENTITY_TYPES_TABLE}").fetchall()
        self._data = {str(row["concept_key"]): str(row["folder_name"]) for row in rows}

    def save(self) -> None:
        """Persist the in-memory registry (full overwrite)."""
        with open_store(self._db_path) as conn:
            conn.execute(f"DELETE FROM {ENTITY_TYPES_TABLE}")
            conn.executemany(
                f"INSERT INTO {ENTITY_TYPES_TABLE}(concept_key, folder_name) VALUES (?, ?)",
                list(self._data.items()),
            )

    def get(self, concept_key: str) -> str | None:
        """Return the registered type for *concept_key*, or None."""
        return self._data.get(concept_key)

    def record(self, concept_key: str, entity_type: str) -> bool:
        """Record *entity_type* for *concept_key*.

        Returns True when the type changed (caller may want to log
        regressions), False when it was already set to the same value
        or unset.

        Empty/None keys or types are ignored — the model occasionally
        returns blank ``type`` fields and those should not pollute the
        registry.
        """
        if not concept_key or not entity_type:
            return False
        existing = self._data.get(concept_key)
        if existing == entity_type:
            return False
        self._data[concept_key] = entity_type
        return existing is not None  # True = override

    def all(self) -> dict[str, str]:
        """Snapshot of the registry (copy)."""
        return dict(self._data)

    def __len__(self) -> int:
        return len(self._data)
