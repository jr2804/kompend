# Multi-Level Caching in KNOX

## Overview

KNOX implements a multi-level, content-addressed caching system that makes the
pipeline fully incremental: adding a single source file only re-processes the
minimum necessary work. Every tier output is cached with full dependency tracking —
if nothing changed, everything is skipped.

## Architecture

```
                          ┌─────────────────────────────────────┐
                          │          CacheStore                  │
                          │  get(layer, deps) → value | None    │
                          │  set(layer, deps, value)             │
                          │  invalidate(layer | None)            │
                          │  has(layer, deps) → bool             │
                          └─────────────────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
        ┌──────────┐        ┌──────────┐        ┌──────────┐
        │  chunk/  │        │ resolve/ │        │ extract/ │
        │ *.pkl    │        │ *.pkl    │        │ *.pkl    │
        └──────────┘        └──────────┘        └──────────┘
              ▼                    ▼                    ▼
        ┌──────────┐        ┌──────────┐        ┌──────────┐
        │generate/ │        │ overview/│        │ (future) │
        │ *.pkl    │        │ *.pkl    │        │          │
        └──────────┘        └──────────┘        └──────────┘
```

## Cache Layers (bottom-up)

### Layer 1: Parse Cache (Tier 1, *existing*)
- **Location**: `.knox/cache/<sha256-prefix>.json`
- **Key**: SHA-256 of source file bytes + `image_mode`
- **Deps**: none (file content is the input)
- **Value**: `ParsedDocument` (JSON-serialized)
- **Inherited**: was already implemented before this caching work

### Layer 2: Chunk Cache (Tier 1, *new*)
- **Location**: `.knox/cache/chunk/<sha256-key>.pkl`
- **Key**: `sha256(source_file_bytes)` + `chunk_size`
- **Deps**: `{"source": <sha256>, "chunk_size": "<n>"}`
- **Value**: `list[Chunk]`
- **Hooks**: `WikiPipeline._process_one()` in `pipeline.py`
- **On hit**: skips `chunker.chunk()` entirely

### Layer 3: Resolution Cache (Tier 2, *new*)
- **Location**: `.knox/cache/resolve/<sha256-key>.pkl`
- **Key**: `sha256(chunk.text)` + `vlm_model`
- **Deps**: `{"chunk_text": <sha256>, "vlm_model": "<model>"}`
- **Value**: `ResolvedChunk`
- **Hooks**: `WikiPipeline._resolve_chunks()` in `pipeline.py`
- **On hit**: skips VLM enrichment for that chunk

### Layer 4: Extraction Cache (Tier 3 Phase 1, *new*)
- **Location**: `.knox/cache/extract/<sha256-key>.pkl`
- **Key**: `sha256(source_text)` + `model` + `extract_budget`
- **Deps**: `{"source_text": <sha256>, "model": "<model>", "extract_budget": "<n>"}`
- **Value**: `Extraction` (summary, entities, concepts, tags)
- **Hooks**: `TwoPhaseCompiler._extract_one()` in `two_phase.py`
- **On hit**: skips LLM extraction call for that source

### Layer 5: Generation Cache (Tier 3 Phase 2, *new*)
- **Location**: `.knox/cache/generate/<sha256-key>.pkl`
- **Key**: `sha256(all_extractions)` + `model` + `gen_source_segment` + `concepts_per_batch`
- **Deps**: `{"extractions": <sha256>, "model": "<model>", "gen_source_segment": "<n>", "concepts_per_batch": "<n>"}`
- **Value**: `CompiledWiki` (nodes, edges, toc)
- **Hooks**: `TwoPhaseCompiler.compile()` in `two_phase.py`
- **On hit**: skips all LLM generation calls; returns cached wiki directly

### Layer 6: Overview Cache (post-compile, *new*)
- **Location**: `.knox/cache/overview/<sha256-key>.pkl`
- **Key**: `sha256(all_node_contents)` + `model` + `overview_budget`
- **Deps**: `{"source_texts": <sha256>, "model": "<model>", "overview_budget": "<n>"}`
- **Value**: `str` (the overview markdown)
- **Hooks**: `write_overview()` in `wiki_index.py`
- **On hit**: skips LLM overview synthesis, reads cached markdown

## Dependency Tracking

Every cache entry stores a `deps` dict mapping dependency names to their
content hashes *at the time the entry was created*. On lookup, the caller
provides the **current** hashes of all dependencies. If any stored hash
differs from the current one, the entry is treated as stale and a cache
miss is returned.

This means:
- **File change**: source file modified → `sha256(source_bytes)` changes →
  chunk, resolve, extract caches all miss automatically
- **Config change**: `chunk_size` changed → `chunk_size` in deps changes →
  chunk cache miss; downstream caches still hit if their deps unchanged
- **Model change**: `default_model` changed → extract/generation cache miss

## Invalidation & Bypass

| Control | Mechanism | Effect |
|---------|-----------|--------|
| `--no-cache` | `PipelineConfig.no_cache` | Bypasses all cache reads (still writes) |
| `--force` | `PipelineConfig.force` | Calls `CacheStore.invalidate()` at pipeline start, then re-processes everything |
| Per-layer | `KNOX_CACHE_DISABLE=chunk,resolve` | Future — env var parsed after tier state work |

## Tier State (Complementary)

The `TierStateStore` (`.knox/compile_state.json`) tracks the highest
completed tier per source file. It works alongside the CacheStore:

- **Tier 1**: recorded after chunking
- **Tier 2**: recorded after resolution
- **Tier 3**: recorded after compilation
- **Tier 4**: recorded after output

This provides a "nothing to do" fast path: if all sources have
`current_tier >= N` and all cache dependencies match, the tier can be
skipped without cache lookups.

## Implementation Details

### CacheStore (`src/knox/core/cache_store.py`)

```python
class CacheStore:
    def get(self, layer: str, deps: dict[str, str]) -> Any | None
    def set(self, layer: str, deps: dict[str, str], value: Any) -> None
    def has(self, layer: str, deps: dict[str, str]) -> bool
    def invalidate(self, layer: str | None = None) -> None
```

- **Storage format**: Pickle files (`.pkl`) with atomic write-to-temp-then-rename
- **Key derivation**: `sha256(json.dumps(deps, sort_keys=True))[:16]`
- **Cache directory**: `<output_dir>/.knox/cache/<layer>/<key>.pkl`
- **Thread safety**: Different keys → different files → no contention;
  atomic writes prevent partial reads

### Make Cache Key (`src/knox/core/cache_store.py`)

```python
def make_cache_key(deps: dict[str, str]) -> str:
    """16-char hex key from sorted dependency dict."""
```

Used externally by callers that need to reference cache entries without
loading them.

## Commits

| Commit | Layer | Description |
|--------|-------|-------------|
| `02167c1` | Foundation | CacheStore class + 17 unit tests |
| `9824ae3` | Chunk, Resolve, Extract | Per-source cache wiring in pipeline.py and two_phase.py |
| `c60b569` | Generate, Overview | Generation cache + overview cache |
| `b23c68b` | Tier state + invalidation | Tier state updates, force invalidation |

## Future Optimizations

1. **Per-concept generation cache**: track which sources contribute to each
   concept; only regenerate concepts whose source signature changed
2. **SQLite backend**: replace pickle files with SQLite for better concurrent
   access and lower overhead
3. **Cache compaction**: periodic cleanup of orphaned entries
4. **Memory cache**: LRU cache in front of disk cache for hot paths
