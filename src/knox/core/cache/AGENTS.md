# DOX — knox.core.cache

<!-- on entry: loaded when agent enters src/knox/core/cache -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Content-addressed JSON cache with dependency validation. Serializes
dataclasses, `Path`, `StrEnum`, `bytes`, and `tuple` via custom JSON codecs.

## Modules

| Module | Contents |
|--------|----------|
| `store.py` | `CacheStore`, `make_cache_key`, `_encode_json`, `_decode_json` |

## Rules

- Import: `from knox.core.cache.store import CacheStore`.
- Depends on `knox.core.models` (for dataclass serialization) and
  `knox.core.utils.time` (for cache timestamps).
- Different inputs write to different cache files; writes are atomic
  (temp-then-rename). Thread-safe by design.
