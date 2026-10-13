# DOX — kompend.core.models

<!-- on entry: loaded when agent enters src/kompend/core/models -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Canonical data models and serialization contracts for the KOMPEND pipeline.
Every data structure that crosses a tier boundary or a file boundary is
defined here. Import from the package root: `from kompend.core.models import Chunk`.

## Structure

| Module          | Contents                                                                                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enums.py`      | StrEnum key contracts (`MetadataField`, `WikiField`, `ChunkField`, `ChunkNodeField`, `ChunkTreeField`, `StructuredField`), `ElementType`, `ATOMIC_TYPES`, `ChunkStrategy` |
| `elements.py`   | `DocumentElement`, `ParsedDocument` (Tier 1 output)                                                                                                                       |
| `chunks.py`     | `Chunk`, `ResolvedChunk` (Tier 2–3 output)                                                                                                                                |
| `structured.py` | `EquationVariable`, `StructuredTable`, `StructuredEquation`, `StructuredFigure`, `StructuredData`, serialization helpers                                                  |
| `trees.py`      | `ChunkNode`, `ChunkTree`, `CHUNK_TREE_CACHE_VERSION`                                                                                                                      |
| `wiki.py`       | `WikiNode`, `CompiledWiki`, `CrossReferenceIssue`, `CrossReferenceResult`, `ExtractionStats`, `load_cached_wiki` (Tier 4 output)                                             |
| `citations.py`  | `Citation`, `extract_citations`, `merge_duplicate_citations`, `LIFECYCLE_*` constants                                                                                     |
| `references.py` | `Reference`, `Author`, `ReferenceExtractionOutput` — bibliographic refs (CSL JSON-aligned)                                                                                |
| `__init__.py`   | Public re-exports — import from here, not from submodules                                                                                                                 |

## Rules

- **Import from the package root.** Never import from `kompend.core.models.chunks`
  etc. — use `from kompend.core.models import Chunk`.
- **Every dict key is a StrEnum member.** See `src/kompend/AGENTS.md`:
  Antipattern 5 (hard-coded schema/field-name strings).
- **Serialization methods live on the dataclass.** Each model has `to_dict()`
  / `from_dict()` using enum-backed keys — no standalone helpers.
- **No text-in-code.** See parent doc.
- `enums.py` has zero internal dependencies; all other modules import from it.
