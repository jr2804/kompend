"""Data models for the KOMPEND pipeline.

Public API re-exported from the split submodules. Import from
``kompend.core.models`` (the package root) — the submodule layout is an
implementation detail.
"""

from __future__ import annotations

from kompend.core.models.attachments import Attachment, AttachmentManifest
from kompend.core.models.chunks import Chunk, ImageRef, ResolvedChunk, image_source_key, images_for_provenance
from kompend.core.models.citations import (
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
from kompend.core.models.elements import DocumentElement, ParsedDocument
from kompend.core.models.enums import (
    ChunkField,
    ChunkNodeField,
    ChunkStrategy,
    ChunkTreeField,
    ElementType,
    MetadataField,
    StructuredField,
    WikiField,
)
from kompend.core.models.references import (
    Author,
    Reference,
    ReferenceExtractionOutput,
    ReferenceResolutionOutput,
)
from kompend.core.models.structured import (
    EquationVariable,
    StructuredData,
    StructuredEquation,
    StructuredFigure,
    StructuredTable,
)
from kompend.core.models.trees import CHUNK_TREE_CACHE_VERSION, ChunkNode, ChunkTree
from kompend.core.models.wiki import (
    CompiledWiki,
    CrossReferenceIssue,
    CrossReferenceResult,
    ExtractionStats,
    GraphEdge,
    TocEntry,
    WikiNode,
    load_cached_wiki,
    load_wiki_from_files,
)

__all__ = [
    "CHUNK_TREE_CACHE_VERSION",
    "LIFECYCLE_ARCHIVED",
    "LIFECYCLE_DRAFT",
    "LIFECYCLE_REVIEWED",
    "LIFECYCLE_STALE",
    "LIFECYCLE_VERIFIED",
    "VALID_LIFECYCLE_STATES",
    "Attachment",
    "AttachmentManifest",
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
    "ExtractionStats",
    "GraphEdge",
    "ImageRef",
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
    "TocEntry",
    "WikiField",
    "WikiNode",
    "extract_citations",
    "image_source_key",
    "images_for_provenance",
    "load_cached_wiki",
    "load_wiki_from_files",
    "merge_duplicate_citations",
]
