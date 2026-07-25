"""Data models for the KNOX pipeline.

Public API re-exported from the split submodules. Import from
``knox.core.models`` (the package root) — the submodule layout is an
implementation detail.
"""

from __future__ import annotations

from knox.core.models.chunks import Chunk, ResolvedChunk
from knox.core.models.citations import (
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_DRAFT,
    LIFECYCLE_REVIEWED,
    LIFECYCLE_STALE,
    LIFECYCLE_VERIFIED,
    VALID_LIFECYCLE_STATES,
    Citation,
    extract_citations,
    merge_duplicate_citations,
)
from knox.core.models.elements import DocumentElement, ParsedDocument
from knox.core.models.enums import (
    ATOMIC_TYPES,
    ChunkField,
    ChunkNodeField,
    ChunkStrategy,
    ChunkTreeField,
    ElementType,
    MetadataField,
    StructuredField,
    WikiField,
)
from knox.core.models.references import (
    Author,
    Reference,
    ReferenceExtractionOutput,
    ReferenceResolutionOutput,
)
from knox.core.models.structured import (
    EquationVariable,
    StructuredData,
    StructuredEquation,
    StructuredFigure,
    StructuredTable,
)
from knox.core.models.trees import CHUNK_TREE_CACHE_VERSION, ChunkNode, ChunkTree
from knox.core.models.wiki import (
    CompiledWiki,
    CrossReferenceIssue,
    CrossReferenceResult,
    WikiNode,
    load_cached_wiki,
)

__all__ = [
    "ATOMIC_TYPES",
    "CHUNK_TREE_CACHE_VERSION",
    "LIFECYCLE_ARCHIVED",
    "LIFECYCLE_DRAFT",
    "LIFECYCLE_REVIEWED",
    "LIFECYCLE_STALE",
    "LIFECYCLE_VERIFIED",
    "VALID_LIFECYCLE_STATES",
    "Author",
    "Chunk",
    "ChunkField",
    "ChunkNode",
    "ChunkNodeField",
    "ChunkStrategy",
    "ChunkTree",
    "ChunkTreeField",
    "Citation",
    "CompiledWiki",
    "CrossReferenceIssue",
    "CrossReferenceResult",
    "DocumentElement",
    "ElementType",
    "EquationVariable",
    "MetadataField",
    "ParsedDocument",
    "Reference",
    "ReferenceExtractionOutput",
    "ReferenceResolutionOutput",
    "ResolvedChunk",
    "StructuredData",
    "StructuredEquation",
    "StructuredField",
    "StructuredFigure",
    "StructuredTable",
    "WikiField",
    "WikiNode",
    "extract_citations",
    "load_cached_wiki",
    "merge_duplicate_citations",
]
