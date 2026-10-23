# Rationale archive (2026-10-23)

Merged out of `.agents/HISTORY.md` to stay inside the `.agents/` line budget.
Each entry records a decision whose rationale is costly to rediscover.

## Why text-in-code is banned

Prompts embedded in Python were undiscoverable, unversionable, and invisible to
non-developers. All prompt text now lives in Jinja2 templates or pack TOML. The
remaining legacy violations are tracked in beads — `bd list`, not a table here.

## Why litellm was removed

`c8055c0`. After moving to pydantic-ai's native `infer_model()` plus the
models.dev catalog, litellm was 40+ transitive dependencies and added latency
around a unified `build_model()` path. Removed.

## Why heavy imports are lazy-loaded (SPEC-0001)

`kompend --help` took ~80s because `register_all_commands()` imported every command
module, each transitively pulling docling/torch/pydantic-ai/MCP. `lazy.attach()`
in five package `__init__.py` files, plus the `get_backend()` factory for
function-body-only classes, took it to 11s (−86%). No `_LAZY` dicts, no
`__getattr__` fallbacks.

## Why `compile.embedding.dim` was removed

The `dim = 384` fallback shaped only the empty-text dedup vector while non-empty
fingerprints used `provider.embed()`'s real dimension, so a mismatch crashed
`EmbeddingMatcher.score(..., strict=True)` — and `make_provider()` ignored the
hint anyway. Both zvec stores had the same literal; all are gone and dim always
resolves from `provider.dim`, lazily probed. Stores rebuild when the stored
dimension differs from the current model's.

## Why `_ENV_LIST_VARS` is schema-derived

The hand-listed `KOMP_*` names drifted from the schema (splitting a `str` field
into a list, listing a dead var). It now derives from list-typed config-map
fields. Separately, `KOMP_CHARS_PER_TOKEN` was a dead env var — the real alias is
`KOMP_LLM_DEFAULT_CHARS_PER_TOKEN` and the value already flows via
`default_chars_per_token`.

## Fixed CLI violations archive (moved from `src/kompend/cli/AGENTS.md`)

| Fixed | Was |
|---|---|
| `_build_router()` -> `build_default_router()` in `kompend.preprocessing.parsers._registry` | Parser construction in CLI |
| `_load_wiki()` -> `load_cached_wiki()` in `kompend.core.models` | JSON deserialization in CLI |
| `_build_vector_index()` -> `build_vector_index()` in `kompend.rag` | RAG business logic in CLI |
| `status.py` now uses `find_project_config()` | Hardcoded `.kompend/config.yaml` path |
| `SUPPORTED_EXTENSIONS` -> `get_supported_extensions()` querying parser registry | Duplicated extension set |

## Why `global` is gone in favour of `functools.cache`

The last three `global` statements (otel one-shot guard, verbosity toggle, pack
registry) and the hand-rolled memo dicts (`_VISION_CACHE`, `_SNAPSHOT`,
`_loader_instance`) became `@cache` getters - memoization is the state, and
tests reset via `getter.cache_clear()` instead of monkeypatching private module
globals. Boolean env reads/writes are centralized in `get_flag()`/`set_flag()`
(`kompend.core.env`); the antipattern checker's global allowlist is now empty.
