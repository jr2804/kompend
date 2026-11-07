"""Stage-side parse-cache adapter: ParsedDocument and chunk-tree cache I/O.

The DB layer lives next door in :mod:`kompend.core.pipeline.parse_cache`; this
module holds the half that needs the pipeline (vault root, rel paths, parser
identity, the chunk-cache dependency set).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from kompend.core.models import (
    ChunkTree,
    DocumentElement,
    ElementType,
    MetadataField,
    ParsedDocument,
    WikiField,
)
from kompend.core.models.elements import element_text
from kompend.core.pipeline.ingest import (
    PARSE_CACHE_PARSER_KEY,
    preserve_cached_metadata,
)
from kompend.core.pipeline.parse_cache import (
    cache_key,
    content_hash_prefix,
    parse_cache_is_current,
    read_document,
    source_key,
    write_document,
)
from kompend.core.pipeline.runtime import PipelineRuntime
from kompend.core.store import vault_db_path
from kompend.core.types import ReasonCode
from kompend.core.verbosity import is_verbose
from kompend.logging import get_logger
from kompend.preprocessing.parsers.base import scrub_document
from kompend.types import JSONDict

logger = get_logger(__name__)


def _try_load_parsed_cache(pipeline: PipelineRuntime, source_file: Path, stem: str, content_hash: str, use_cache: bool) -> ParsedDocument | None:
    """Try to load a parsed document from cache. Returns *None* on miss."""
    if not use_cache:
        return None
    parsed = _load_cache(pipeline, source_file)
    if parsed is not None:
        return parsed
    if pipeline.config.force:
        return None
    cached_entry = pipeline.state.entries.get(source_key(stem, content_hash))
    if cached_entry and cached_entry.content_hash == content_hash and cached_entry.current_tier >= 1:
        return _load_cache(pipeline, source_file)
    return None


def _load_cache(pipeline: PipelineRuntime, source_file: Path) -> ParsedDocument | None:
    """Load the cached ParsedDocument for *source_file*, or *None* on a miss.

    The cache key is ``<stem>-<hash8>``, so the entry is content-addressed and
    a source may be moved or regrouped under ``sources/`` without losing it.
    The stored ``source_file`` is therefore only a historical note: a hit is
    rebound to *source_file* (the path this run looked the entry up for) so
    citations, tags and the state store never resurrect a path that no longer
    exists. See ``tests/unit/test_ingest_reparse.py``.

    Content alone does not make an entry current: the parser identity decides
    whether this run would produce the same document. A changed backend set, or
    a changed routing input (judge, preference, backend options), makes the
    stored entry a miss — otherwise a vault would keep serving parses that the
    configuration no longer asks for.
    """
    db_path = vault_db_path(Path(pipeline.base_directory))
    content_hash = content_hash_prefix(source_file.read_bytes())
    key = cache_key(source_file, content_hash)
    data = read_document(db_path, key)
    if data is None:
        return None
    if not parse_cache_is_current(data, parse_identity(pipeline)):
        logger.info("Re-parsing %s: cached parse is not from the current parser", source_file.name)
        return None
    try:
        elements = [
            DocumentElement(
                type=ElementType(el["type"]),
                content=el["content"],
                page=el.get("page"),
                reading_order=el.get(MetadataField.READING_ORDER),
                heading_level=el.get("heading_level"),
                metadata=el.get("metadata", {}),
            )
            for el in cast("list[dict[str, Any]]", data.get("elements") or [])
        ]
        # scrub_document: caches written before surrogate scrubbing carry
        # lone \\udXXX escapes that json.loads restores as surrogate chars.
        return scrub_document(
            ParsedDocument(
                source_file=source_file,
                elements=elements,
                title=cast("str", data.get("title", "")),
                author=cast("str", data.get("author", "")),
                language=cast("str", data.get("language", "en")),
                page_count=cast("int", data.get("page_count", 0)),
                metadata=cast("dict[str, Any]", data.get(WikiField.METADATA) or {}),
            )
        )
    except (KeyError, ValueError, TypeError) as exc:
        # Corrupt entry = miss: re-parse. Warn so it's visible.
        logger.warning("Corrupt parse cache for %s — reparsing: %s", key, exc, exc_info=is_verbose(), extra={"reason_code": ReasonCode.CORRUPT_CACHE})
        return None


def _save_cache(pipeline: PipelineRuntime, source_file: Path, doc: ParsedDocument) -> None:
    """Write a ParsedDocument to the store, stamped with the parser identity."""
    db_path = vault_db_path(Path(pipeline.base_directory))
    content_hash = content_hash_prefix(source_file.read_bytes())
    key = cache_key(source_file, content_hash)
    try:
        # Preserve user-set tags/attributes across the overwrite.
        preserve_cached_metadata(doc, read_document(db_path, key))
        data = {
            str(MetadataField.SOURCE_FILE): pipeline.rel_file(doc.source_file),
            PARSE_CACHE_PARSER_KEY: parse_identity(pipeline),
            "title": doc.title,
            "author": doc.author,
            "language": doc.language,
            "page_count": doc.page_count,
            "metadata": doc.metadata,
            "elements": [
                {
                    "type": str(el.type),
                    # element_text, not str(): bytes would land in the cache (and
                    # every later read of it) as a repr blob.
                    "content": el.content if isinstance(el.content, list) else element_text(el.content),
                    "page": el.page,
                    MetadataField.READING_ORDER: el.reading_order,
                    "heading_level": el.heading_level,
                    "metadata": el.metadata,
                }
                for el in doc.elements
            ],
        }
        write_document(db_path, key, cast("JSONDict", data))
    except Exception as exc:
        logger.debug("Cache write failed for %s — ignoring: %s", key, exc, exc_info=is_verbose())


def parse_identity(pipeline: PipelineRuntime) -> str:
    """The identity of the parse this run would produce.

    ``DocumentParser.backend_identity()`` composes both halves: the backends that
    can read here, and the routing inputs that change what they emit (judge tier,
    preference, per-backend options, upstream version). The parse cache stores it
    and compares it on read, so a configuration change re-parses instead of being
    silently ignored for every document already in the cache.
    """
    parser = pipeline.parser
    return parser.backend_identity() if parser is not None else ""


def _try_load_chunk_cache(pipeline: PipelineRuntime, source_file: Path, content_hash: str, use_cache: bool) -> ChunkTree | None:
    """Try to load a chunked tree from cache. Returns *None* on miss.

    The tree is rebound to *source_file*: the chunk-cache identity is the
    content hash, so a source may be moved or regrouped under ``sources/``
    and still hit; the path it carries must be the one this run is compiling.
    """
    if not (use_cache and content_hash):
        return None
    chunk_deps = _chunk_cache_deps(pipeline, content_hash)
    cached = pipeline.cache_store.get_dict("chunk", chunk_deps)
    if cached is None:
        return None
    stem = source_file.stem
    pipeline.state_store.record(pipeline.state, stem, Path(pipeline.rel_file(source_file)), content_hash, tier=1)
    tree = ChunkTree.from_dict(cached)
    tree.source_file = source_file
    pipeline.apply_tags_to_tree(tree, source_file)
    return tree


def _chunk_cache_deps(pipeline: PipelineRuntime, content_hash: str) -> dict[str, str]:
    """Cache identity for a chunked tree.

    ``source`` + chunker size identify the chunking.
    """
    return {
        "source": content_hash,
        "chunk_effective_max": str(getattr(pipeline.chunker, "_effective_max", 0)),
        "contextual": "1" if (pipeline.config.search.contextual.enabled and pipeline.enricher is not None) else "0",
    }
