from __future__ import annotations

import sqlite3
from enum import StrEnum
from pathlib import Path
from typing import Any

from knox.core.models import CompiledWiki, MetadataField
from knox.logging import get_logger

logger = get_logger(__name__)


class FtsField(StrEnum):
    """Column and result-dict key names for the FTS5 chunk index.

    Single source of truth for column names used in CREATE TABLE,
    INSERT, SELECT, and the result dict.
    """

    ID = "id"
    ARTICLE_ID = "article_id"
    TITLE = "title"
    CONTENT = "content"
    HEADING = "heading"


_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS chunks (
    {FtsField.ID} TEXT PRIMARY KEY,
    {FtsField.ARTICLE_ID} TEXT NOT NULL,
    {FtsField.TITLE} TEXT NOT NULL,
    {FtsField.CONTENT} TEXT NOT NULL,
    {MetadataField.SOURCE_FILE} TEXT,
    {FtsField.HEADING} TEXT
)
"""

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    title, content, heading,
    content=chunks,
    content_rowid=rowid
)
"""


def build_chunk_index(wiki: CompiledWiki, index_file: Path) -> None:
    index_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(index_file))
    try:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_FTS)
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM chunks_fts")

        for node in wiki.nodes:
            source = ""
            if node.metadata and MetadataField.SOURCE in node.metadata:
                source = str(node.metadata[MetadataField.SOURCE])
            _cols = f"{FtsField.ID}, {FtsField.ARTICLE_ID}, {FtsField.TITLE}, {FtsField.CONTENT}, {MetadataField.SOURCE_FILE}, {FtsField.HEADING}"
            conn.execute(
                f"INSERT INTO chunks ({_cols}) VALUES (?, ?, ?, ?, ?, ?)",  # noqa: S608
                (node.id, node.id, node.title, node.content or "", source, ""),
            )

        conn.commit()
        row_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        logger.info("Indexed %d chunks in %s", row_count, index_file)
    finally:
        conn.close()


def search_fts(query: str, index_file: Path, top_k: int = 5) -> list[dict[str, Any]]:
    if not index_file.exists():
        return []

    conn = sqlite3.connect(str(index_file))
    try:
        clean = query.replace('"', '""')
        _scols = f"c.{FtsField.ID}, c.{FtsField.ARTICLE_ID}, c.{FtsField.TITLE}, c.{FtsField.CONTENT}, c.{MetadataField.SOURCE_FILE}"
        rows = conn.execute(
            f"SELECT {_scols}, "  # noqa: S608
            "rank FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (clean, top_k),
        ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    str(FtsField.ID): row[0],
                    str(FtsField.ARTICLE_ID): row[1],
                    str(FtsField.TITLE): row[2],
                    str(FtsField.CONTENT): row[3],
                    str(MetadataField.SOURCE_FILE): row[4],
                    "score": 1.0 / (1.0 + abs(row[5])) if row[5] is not None else 0.0,
                }
            )
        return results
    finally:
        conn.close()
