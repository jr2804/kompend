"""StrEnum contracts and element-type definitions for KNOX data models.

Every dict key, discriminator, or schema column name that crosses a file
boundary lives here as a ``StrEnum`` member — never as a bare literal.
"""

from __future__ import annotations

from enum import StrEnum


class MetadataField(StrEnum):
    """Canonical keys for ``WikiNode.metadata`` and provenance data dicts.

    Every metadata / source-file / provenance key that crosses file boundaries
    must live here, not as a bare literal.
    """

    SOURCE = "source"
    SOURCE_FILE = "source_file"
    SOURCE_PATH = "source_path"
    SOURCE_COUNT = "source_count"
    SOURCE_NAMES = "source_names"
    HEADING_CONTEXT = "heading_context"
    READING_ORDER = "reading_order"
    PROVENANCE = "provenance"
    IMAGE_PATH = "image_path"
    IMAGE_BASE64 = "image_base64"
    TAGS = "tags"


class WikiField(StrEnum):
    """Dict key contract for ``CompiledWiki`` JSON serialization.

    Keys written by :meth:`CompiledWiki.to_dict` (or built by pipeline) and
    read by :func:`CompiledWiki.from_dict` and ``web/server.py``.
    """

    NODES = "nodes"
    EDGES = "edges"
    TOC = "toc"
    QUALITY_SCORE = "quality_score"
    ITERATION = "iteration"
    CHUNK_COUNT = "chunk_count"
    DURATION_SECONDS = "duration_seconds"
    LLM_USAGE = "llm_usage"
    LLM_MODEL = "llm_model"
    LLM_DEGRADATION = "llm_degradation"
    TIER_TIMINGS = "tier_timings"
    METADATA = "metadata"


class ChunkField(StrEnum):
    """Dict key contract for ``Chunk`` JSON serialization."""

    ID = "id"
    TEXT = "text"
    ELEMENT_TYPES = "element_types"
    SOURCE_ELEMENTS = "source_elements"
    SOURCE_FILE = "source_file"
    HEADING_CONTEXT = "heading_context"
    TOKEN_COUNT = "token_count"  # noqa: S105
    READING_ORDER = "reading_order"
    METADATA = "metadata"


class ChunkNodeField(StrEnum):
    """Dict key contract for ``ChunkNode`` JSON serialization."""

    ID = "id"
    PARENT_ID = "parent_id"
    CHILDREN_IDS = "children_ids"
    DEPTH = "depth"
    HEADING_PATH = "heading_path"
    CHUNK = "chunk"
    STRUCTURED = "structured"
    SUMMARY = "summary"
    TOKEN_COUNT = "token_count"  # noqa: S105


class ChunkTreeField(StrEnum):
    """Dict key contract for ``ChunkTree`` JSON serialization."""

    VERSION = "version"
    ROOT_ID = "root_id"
    NODES = "nodes"
    SOURCE_FILE = "source_file"
    STRATEGY = "strategy"


class StructuredField(StrEnum):
    """Dict key contract for ``StructuredData`` JSON serialization.

    Shared discriminator key plus per-variant keys. Each variant reuses
    ``_TYPE`` for its ``_type`` discriminator and inherits common fields
    like ``caption``.
    """

    _TYPE = "_type"
    TYPE_TABLE = "table"
    TYPE_EQUATION = "equation"
    TYPE_FIGURE = "figure"
    COLUMNS = "columns"
    ROWS = "rows"
    CAPTION = "caption"
    LATEX = "latex"
    VARIABLES = "variables"
    NAME = "name"
    DEFINITION = "definition"
    UNITS = "units"
    ALT_TEXT = "alt_text"
    IMAGE_REF = "image_ref"
    DESCRIPTION = "description"


class ElementType(StrEnum):
    """Document element types for structure-preserving extraction.

    StrEnum serializes to plain strings in JSON: ``"text"``, ``"table"``, etc.
    """

    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"
    LIST_ITEM = "list_item"
    CODE = "code"
    METADATA = "metadata"


ATOMIC_TYPES: frozenset[ElementType] = frozenset({ElementType.TABLE, ElementType.IMAGE, ElementType.FORMULA, ElementType.CODE})
"""Element types that must never be split during chunking."""


class ChunkStrategy(StrEnum):
    """How a document was chunked — determines downstream processing depth.

    - ``SINGLE_PASS``: document fits in context window; trivial tree (root +
      leaves). No summaries needed.
    - ``TWO_LEVEL``: document is <= 8× context window; root + section nodes +
      leaves. Root-level summary generated for navigation.
    - ``MULTI_RESOLUTION``: document is >> 8× context window; full-depth tree
      with summaries at every internal node for multi-resolution navigation.
    """

    SINGLE_PASS = "single_pass"  # noqa: S105
    TWO_LEVEL = "two_level"
    MULTI_RESOLUTION = "multi_resolution"


__all__ = [
    "ATOMIC_TYPES",
    "ChunkField",
    "ChunkNodeField",
    "ChunkStrategy",
    "ChunkTreeField",
    "ElementType",
    "MetadataField",
    "StructuredField",
    "WikiField",
]
