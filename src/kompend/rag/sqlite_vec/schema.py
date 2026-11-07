"""The store's schema — table names, DDL and the fingerprint that decides a rebuild.

Everything here is SQL text and vocabulary, no I/O policy: the connection is
passed in. The tables are ``vec0`` and FTS5 virtual tables, which SQLAlchemy
Core cannot express, so the DDL is hand-written by contract (bead knox-97pg).

The names are module-level so the query, filter and write paths address the
same tables; a literal table name anywhere else is a bug.
"""

# ruff: noqa: S608 — SQL is built from Final[str] table-name constants only.

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


def schema_fingerprint(dim: int) -> str:
    """Identity of the store's schema — a change forces a rebuild."""
    return f"vec0:1:{dim}:{VEC_ITEMS}/{VEC_FTS}/{VEC_TAGS}/{VEC_ATTRIBUTES}/{VEC_ATTR_KEYS}"


def create_schema(conn: sqlite3.Connection, dim: int, model_id: str) -> None:
    """(Re)create the store's tables.

    The index is rebuilt from scratch on every build: the corpus is small,
    and a fresh table cannot carry rows the current wiki no longer has.
    """
    conn.execute(f"DROP TABLE IF EXISTS {VEC_ITEMS}")
    conn.execute(f"DROP TABLE IF EXISTS {VEC_FTS}")
    conn.execute(f"DROP TABLE IF EXISTS {VEC_TAGS}")
    conn.execute(f"DROP TABLE IF EXISTS {VEC_ATTRIBUTES}")
    conn.execute(f"DROP TABLE IF EXISTS {VEC_ATTR_KEYS}")
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE {VEC_ITEMS} USING vec0(
            id              TEXT PRIMARY KEY,
            embedding       FLOAT[{dim}] distance_metric=cosine,
            title           TEXT,
            node_type       TEXT,
            source_file     TEXT,
            content_preview TEXT
        )
        """
    )
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE {VEC_FTS} USING fts5(
            id UNINDEXED,
            title,
            content_preview,
            node_type UNINDEXED,
            source_file UNINDEXED,
            tokenize='unicode61'
        )
        """
    )
    conn.execute(f"CREATE TABLE {VEC_TAGS} (item_id TEXT, tag TEXT, PRIMARY KEY (item_id, tag))")
    conn.execute(f"CREATE TABLE {VEC_ATTRIBUTES} (item_id TEXT, token TEXT, PRIMARY KEY (item_id, token))")
    conn.execute(f"CREATE TABLE {VEC_ATTR_KEYS} (item_id TEXT, key TEXT, PRIMARY KEY (item_id, key))")
    conn.execute(f"CREATE INDEX ix_vec_tags_token ON {VEC_TAGS} (tag)")
    conn.execute(f"CREATE INDEX ix_vec_attributes_token ON {VEC_ATTRIBUTES} (token)")
    conn.execute(f"CREATE INDEX ix_vec_attr_keys_key ON {VEC_ATTR_KEYS} (key)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {VEC_META} (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(f"DELETE FROM {VEC_META}")
    write_meta(conn, dim, model_id)
    conn.commit()


def write_meta(conn: sqlite3.Connection, dim: int, model_id: str) -> None:
    """Record the store's schema fingerprint."""
    conn.executemany(
        f"INSERT OR REPLACE INTO {VEC_META} (key, value) VALUES (?, ?)",
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
        rows = dict(conn.execute(f"SELECT key, value FROM {VEC_META}").fetchall())
    except sqlite3.Error:
        return False
    return rows.get("backend") == BACKEND and rows.get("dim") == str(dim) and rows.get("schema") == schema_fingerprint(dim)
