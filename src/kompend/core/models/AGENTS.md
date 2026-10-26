# DOX — kompend.core.models

<!-- on entry: loaded when agent enters src/kompend/core/models -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Canonical data models and serialization contracts for the kompend pipeline.
Every data structure that crosses a tier boundary or a file boundary is
defined here. Import from the package root: `from kompend.core.models import Chunk`.

## Structure

| Module          | Contents                                                                                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enums.py`      | StrEnum key contracts (`MetadataField`, `WikiField`, `ChunkField`, `ChunkNodeField`, `ChunkTreeField`, `StructuredField`), `ElementType`, `ATOMIC_TYPES`, `ChunkStrategy`, `NodeType`, `OkfField`, `EdgeType` |
| `kinds.py`      | `KIND_FOLDERS` (NodeType value → vault folder) and `KindFolder` — lives here, not `core/paths.py`, because paths → models → paths would cycle |
| `elements.py`   | `DocumentElement`, `ParsedDocument` (Tier 1 output)                                                                                                                       |
| `chunks.py`     | `Chunk`, `ResolvedChunk` (Tier 2–3 output)                                                                                                                                |
| `structured.py` | `EquationVariable`, `StructuredTable`, `StructuredEquation`, `StructuredFigure`, `StructuredData`, serialization helpers                                                  |
| `trees.py`      | `ChunkNode`, `ChunkTree`, `CHUNK_TREE_CACHE_VERSION`                                                                                                                      |
| `wiki.py`       | `WikiNode`, `CompiledWiki`, `CrossReferenceIssue`, `CrossReferenceResult`, `ExtractionStats`, `load_cached_wiki` (Tier 4 output)                                             |
| `metadata.py`   | Narrowing helpers for `MetadataValue` — `meta_str`, `meta_int`, `meta_float`, `meta_str_list`, `meta_str_dict`, `meta_str_list_dict`, `meta_dict`, `meta_dict_list`, `meta_list_dict`, `meta_float_dict` |
| `citations.py`  | `Citation`, `extract_citations`, `merge_duplicate_citations`, `LIFECYCLE_*` constants                                                                                     |
| `references.py` | `Reference`, `Author`, `ReferenceExtractionOutput` — bibliographic refs (CSL JSON-aligned)                                                                                |
| `__init__.py`   | Public re-exports — import from here, not from submodules                                                                                                                 |

## Rules

- **Import from the package root.** Never import from `kompend.core.models.chunks`
  etc. — use `from kompend.core.models import Chunk`.
- **Every dict key is a StrEnum member.** See `src/kompend/AGENTS.md`:
  Antipattern 5 (hard-coded schema/field-name strings).
- **One vocabulary per domain.** `NodeType` is the only node-type vocabulary
  (`node_type` values, OKF `type` labels — `node_type_from_okf_type()` maps an
  OKF label back, unknown → `NodeType.CONCEPT`); `OkfField` holds the OKF
  frontmatter keys. No bare `"concept"` / `"type"` literals at a call site.
- **Serialization methods live on the dataclass.** Each model has `to_dict()`
  / `from_dict()` using enum-backed keys — no standalone helpers.
- **Narrow metadata through the helpers.** `WikiNode.metadata` holds
  `MetadataValue`; read it with `meta_*` from `metadata.py` instead of casting
  by hand — they return an empty list/dict or the given default, never raise.
- **No text-in-code.** See parent doc.
- `enums.py` has zero internal dependencies; all other modules import from it.
