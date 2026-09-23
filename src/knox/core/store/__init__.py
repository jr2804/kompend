"""Single non-vector store for a KNOX vault — one SQLite file (``.knox/knox.db``).

Public API:

* :func:`open_store` — context manager that opens the DB, ensures the schema,
  and commits/closes (preferred entry point).
* :func:`connect` — raw connection with PRAGMAs applied.
* :func:`ensure_schema` — create, or clean-cut recreate on a version mismatch.
* :data:`SCHEMA_VERSION`, :data:`TABLE_NAMES` — schema contract.
* :func:`obsolete_artifacts` — legacy flat-file detection (not migration).

The embedding branch (``vectors/``, ``memory.db``) is intentionally separate and
regenerable; this store holds everything else.
"""

from __future__ import annotations

from knox.core.store.connection import connect, ensure_schema, open_store
from knox.core.store.legacy import obsolete_artifacts
from knox.core.store.locate import resolve_base_dir, vault_db_path
from knox.core.store.schema import SCHEMA_VERSION, TABLE_NAMES

__all__ = [
    "SCHEMA_VERSION",
    "TABLE_NAMES",
    "connect",
    "ensure_schema",
    "obsolete_artifacts",
    "open_store",
    "resolve_base_dir",
    "vault_db_path",
]
