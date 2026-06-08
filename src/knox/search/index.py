from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from knox.core.models import CompiledWiki

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    article_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source_file TEXT,
    heading TEXT
)
"""

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    title, content, heading,
    content=chunks,
    content_rowid=rowid
)
"""


def build_chunk_index(wiki: CompiledWiki, index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(index_path))
    try:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_FTS)
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM chunks_fts")

        for node in wiki.nodes:
            source = ""
            if node.metadata and "source" in node.metadata:
                source = str(node.metadata["source"])
            conn.execute(
                "INSERT INTO chunks (id, article_id, title, content, source_file, heading) VALUES (?, ?, ?, ?, ?, ?)",
                (node.id, node.id, node.title, node.content or "", source, ""),
            )

        conn.commit()
        row_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        logger.info("Indexed %d chunks in %s", row_count, index_path)
    finally:
        conn.close()


def search_fts(query: str, index_path: Path, top_k: int = 5) -> list[dict[str, Any]]:
    if not index_path.exists():
        return []

    conn = sqlite3.connect(str(index_path))
    try:
        clean = query.replace('"', '""')
        rows = conn.execute(
            "SELECT c.id, c.article_id, c.title, c.content, c.source_file, "
            "rank FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (clean, top_k),
        ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "id": row[0],
                    "article_id": row[1],
                    "title": row[2],
                    "content": row[3],
                    "source_file": row[4],
                    "score": 1.0 / (1.0 + abs(row[5])) if row[5] is not None else 0.0,
                }
            )
        return results
    finally:
        conn.close()
