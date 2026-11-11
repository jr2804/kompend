"""The store's schema — table names, DDL and the fingerprint that decides a rebuild.

Everything here is SQL text and vocabulary, no I/O policy: the connection is
passed in. The tables are ``vec0`` and FTS5 virtual tables, which SQLAlchemy
Core cannot express, so the DDL is hand-written by contract (bead knox-97pg).

The names are module-level so the query, filter and write paths address the
same tables; a literal table name anywhere else is a bug. Every statement is a
constant template filled by :func:`sql` — no caller assembles SQL by hand,
which is what the module-wide bandit waiver this replaces was hiding.
"""

from __future__ import annotations

import sqlite3
from typing import Final

from kompend.rag.vector_store import EMBED_TEXT_VERSION

#: Identifies this backend in the manifest row and in the store's own meta
#: table. A manifest written by another backend must not validate this store.
BACKEND: Final[str] = "sqlite-vec"

#: The vec0 table holding the vectors and the scalar fields of a node.
VEC_ITEMS: Final[str] = "vec_items"

#: The FTS5 table holding the lexical index (title + content preview). The
#: node type and source file ride along as unindexed columns, so a lexical
#: hit carries the same fields as a vector hit without a join.
VEC_FTS: Final[str] = "vec_fts"

#: Side tables: one row per (node, tag/attribute/key) pair.
VEC_TAGS: Final[str] = "vec_item_tags"
VEC_ATTRIBUTES: Final[str] = "vec_item_attributes"
VEC_ATTR_KEYS: Final[str] = "vec_item_attr_keys"

#: The store's own schema fingerprint — what makes a rebuild necessary.
VEC_META: Final[str] = "vec_meta"

#: The manifest column naming the embedding model behind the vectors.
#: It is a store column, not a node-metadata key — the manifest is not a
#: :class:`~kompend.core.models.MetadataField` carrier — so it carries its own
#: constant instead of borrowing the metadata member.
EMBEDDING_MODEL_COLUMN: Final[str] = "embedding_model"

#: The manifest column naming the backend that wrote the vectors.
BACKEND_COLUMN: Final[str] = "backend"

_DROP_TABLE: Final[str] = "DROP TABLE IF EXISTS {table}"

_CREATE_ITEMS: Final[str] = """
CREATE VIRTUAL TABLE {table} USING vec0(
    id              TEXT PRIMARY KEY,
    embedding       FLOAT[{dim}] distance_metric=cosine,
    title           TEXT,
    node_type       TEXT,
    source_file     TEXT,
    content_preview TEXT
)
"""

_CREATE_FTS: Final[str] = """
CREATE VIRTUAL TABLE {table} USING fts5(
    id UNINDEXED,
    title,
    content_preview,
    node_type UNINDEXED,
    source_file UNINDEXED,
    tokenize='unicode61'
)
"""

_CREATE_TAGS: Final[str] = "CREATE TABLE {table} (item_id TEXT, tag TEXT, PRIMARY KEY (item_id, tag))"
_CREATE_ATTRIBUTES: Final[str] = "CREATE TABLE {table} (item_id TEXT, token TEXT, PRIMARY KEY (item_id, token))"
_CREATE_ATTR_KEYS: Final[str] = "CREATE TABLE {table} (item_id TEXT, key TEXT, PRIMARY KEY (item_id, key))"
_CREATE_INDEX: Final[str] = "CREATE INDEX {index} ON {table} ({column})"
_CREATE_META: Final[str] = "CREATE TABLE IF NOT EXISTS {table} (key TEXT PRIMARY KEY, value TEXT)"
_CLEAR_META: Final[str] = "DELETE FROM {table}"
_WRITE_META: Final[str] = "INSERT OR REPLACE INTO {table} (key, value) VALUES (?, ?)"
_READ_META: Final[str] = "SELECT key, value FROM {table}"


def create_schema(conn: sqlite3.Connection, dim: int, model_id: str) -> None:
    """(Re)create the store's tables.

    The index is rebuilt from scratch on every build: the corpus is small,
    and a fresh table cannot carry rows the current wiki no longer has.
    """
    for table in (VEC_ITEMS, VEC_FTS, VEC_TAGS, VEC_ATTRIBUTES, VEC_ATTR_KEYS):
        conn.execute(sql(_DROP_TABLE, table=table))
    conn.execute(sql(_CREATE_ITEMS, table=VEC_ITEMS, dim=dim))
    conn.execute(sql(_CREATE_FTS, table=VEC_FTS))
    conn.execute(sql(_CREATE_TAGS, table=VEC_TAGS))
    conn.execute(sql(_CREATE_ATTRIBUTES, table=VEC_ATTRIBUTES))
    conn.execute(sql(_CREATE_ATTR_KEYS, table=VEC_ATTR_KEYS))
    conn.execute(sql(_CREATE_INDEX, index="ix_vec_tags_token", table=VEC_TAGS, column="tag"))
    conn.execute(sql(_CREATE_INDEX, index="ix_vec_attributes_token", table=VEC_ATTRIBUTES, column="token"))
    conn.execute(sql(_CREATE_INDEX, index="ix_vec_attr_keys_key", table=VEC_ATTR_KEYS, column="key"))
    conn.execute(sql(_CREATE_META, table=VEC_META))
    conn.execute(sql(_CLEAR_META, table=VEC_META))
    write_meta(conn, dim, model_id)
    conn.commit()


def write_meta(conn: sqlite3.Connection, dim: int, model_id: str) -> None:
    """Record the store's schema fingerprint."""
    conn.executemany(
        sql(_WRITE_META, table=VEC_META),
        [
            ("backend", BACKEND),
            ("dim", str(dim)),
            ("schema", schema_fingerprint(dim)),
            ("embed_version", str(EMBED_TEXT_VERSION)),
            ("model_id", model_id),
        ],
    )


def read_meta(conn: sqlite3.Connection, dim: int) -> bool:
    """True when the store file holds this schema, dimension and backend."""
    try:
        rows = dict(conn.execute(sql(_READ_META, table=VEC_META)).fetchall())
    except sqlite3.Error:
        return False
    return rows.get("backend") == BACKEND and rows.get("dim") == str(dim) and rows.get("schema") == schema_fingerprint(dim)


def sql(template: str, **values: object) -> str:
    """Fill one of this module's SQL templates.

    The virtual tables have no Core object to select from, so the statement
    stays a constant template and the values are substituted in one place.
    The plain-``str`` parameter is the shape a static SQL-injection check can
    follow: table names and numbers only, never user text.
    """
    return template.format(**values)


def schema_fingerprint(dim: int) -> str:
    """Identity of the store's schema — a change forces a rebuild."""
    tables = f"{VEC_ITEMS}/{VEC_FTS}/{VEC_TAGS}/{VEC_ATTRIBUTES}/{VEC_ATTR_KEYS}"
    return f"vec0:1:{dim}:{tables}"
