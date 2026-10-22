# DOX — kompend.core.cache

<!-- on entry: loaded when agent enters src/kompend/core/cache -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Content-addressed JSON cache with dependency validation. Serializes
dataclasses, `Path`, `StrEnum`, `bytes`, and `tuple` via custom JSON codecs.

## Modules

| Module | Contents |
|--------|----------|
| `store.py` | `CacheStore`, `make_cache_key`, `_encode_json`, `_decode_json` |

## Rules

- Import: `from kompend.core.cache.store import CacheStore`.
- A `StrEnum` that reaches the cache must be registered in `_CACHE_ENUMS` —
  the codec tags enums by class name and raises `ValueError` on an unknown
  tag when decoding.
- Depends on `kompend.core.models` (for dataclass serialization) and
  `kompend.core.utils.time` (for cache timestamps).
- Different inputs write to different cache files; writes are atomic
  (temp-then-rename). Thread-safe by design.
