"""SQLAlchemy Core metadata for the ``.kompend/kompend.db`` store.

The ``Table`` objects below are the schema's declared single source of truth
(the typed-store plan, ``.agents/plans/001-typed-store-and-validated-rows.md``):
the DDL is *generated* from this metadata instead of being written twice — once
as structure here, once as hand-written ``CREATE TABLE`` text in ``schema.py``.
The metadata lands table by table; ``tests/unit/test_store.py`` proves each
generated table means exactly what the hand-written DDL means (columns, types,
nullability, defaults, primary keys).

Nullability is copied verbatim, including SQLite's quirk that a bare
``TEXT PRIMARY KEY`` does *not* imply ``NOT NULL`` (``PRAGMA table_info``
reports ``notnull = 0`` for it). SQLAlchemy adds ``NOT NULL`` to primary-key
columns on its own, so the explicit ``nullable=True`` below switches that off
wherever the hand-written DDL declares a bare primary key — a generated schema
that is stricter or looser than the on-disk one is drift, not an improvement.

Defaults are ``server_default`` (SQL rendered into the DDL), never the
Python-side ``default=``: a Python-side default generates no ``DEFAULT``
clause, which is a different schema.
"""

from __future__ import annotations

from typing import Final

from sqlalchemy import Column, Integer, MetaData, PrimaryKeyConstraint, Table, Text
from sqlalchemy.sql import text

#: The store's schema, expressed once. The DDL is generated from this via
#: ``METADATA.create_all(...)`` — never hand-written next to it.
METADATA: Final[MetaData] = MetaData()

#: Parsed-document cache. Keyed by the content discriminator so same-stem files
#: with different content never collide.
documents: Final[Table] = Table(
    "documents",
    METADATA,
    Column("cache_key", Text, primary_key=True, nullable=True),
    Column("source_file", Text, nullable=False),
    Column("parser_fingerprint", Text, nullable=False, server_default=text("''")),
    Column("title", Text, nullable=False, server_default=text("''")),
    Column("author", Text, nullable=False, server_default=text("''")),
    Column("language", Text, nullable=False, server_default=text("'en'")),
    Column("page_count", Integer, nullable=False, server_default=text("0")),
    Column("metadata", Text, nullable=False, server_default=text("'{}'")),
    Column("elements", Text, nullable=False, server_default=text("'[]'")),
)

#: LLM / resolve / generate / summary cache layers (CacheStore). Layer-scoped
#: because the same dependency key can appear under different layers.
extractions: Final[Table] = Table(
    "extractions",
    METADATA,
    Column("layer", Text, nullable=False),
    Column("cache_key", Text, nullable=False),
    Column("deps", Text, nullable=False, server_default=text("'{}'")),
    Column("payload", Text, nullable=False),
    PrimaryKeyConstraint("layer", "cache_key"),
)
