# DOX — kompend.core.pipeline

<!-- on entry: loaded when agent enters src/kompend/core/pipeline -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

5-tier pipeline orchestration: index → chunk → resolve → compile → output.
Contains the orchestrator, pipeline configuration dataclass, tier state
persistence, and standalone ingestion.

## Modules

| Module                | Contents                                                                                 |
| --------------------- | ---------------------------------------------------------------------------------------- |
| `stages/_pipeline.py` | `WikiPipeline` — 5-tier orchestrator, `ProgressCallback`                                 |
| `stages/_factory.py`  | `build_pipeline` — pipeline factory                                                      |
| `stages/_parse.py`    | Tier 1 parse + chunk orchestration                                                       |
| `stages/_index.py`    | Tier 0 source indexing                                                                   |
| `stages/_resolve.py`  | Tier 3 resolution orchestration                                                          |
| `stages/_compile.py`  | Tier 4 compilation + Tier 5 output orchestration                                         |
| `config.py`           | `Tier`, `Compiler`, `ALL_TIERS`, `TIER_LEVEL`, `TIER_DESCRIPTIONS`, `PipelineConfig`     |
| `state.py`            | `TierStateStore`, `TierState`, `SourceState`, `CompileStatus`, `classify_compile_status` |
| `references.py`       | `ReferenceStore` — JSON-backed bibliographic reference persistence                       |
| `ingest.py`           | `ingest_documents`, `parse_cache_name`, `preserve_cached_metadata`, `parse_cache_is_current`, `PARSE_CACHE_PARSER_KEY`, `IngestResult` |
| `parallelism.py`      | `gil_enabled`, `describe_parallelism` — interpreter parallelism facts for `compile.workers` |

## Rules

- Import from submodules directly: `from kompend.core.pipeline.stages._pipeline import WikiPipeline`,
  `from kompend.core.pipeline.config import PipelineConfig`.
- Public re-exports in `__init__.py` preserve backward compatibility —
  `from kompend.core.pipeline.orchestrator import WikiPipeline` still works.
- `config.py` is the public config shape — both CLI and orchestrator import it.
- `_pipeline.py` imports from nearly every other core submodule; this is
  expected for an orchestrator.
- Each stage module keeps its responsibility isolated — stages communicate
  through the `WikiPipeline` state and shared data models.
- The parse cache is parser-aware: entries carry `PARSE_CACHE_PARSER_KEY`
  (from `ParserRouter.fingerprint`) and a mismatched fingerprint is a miss,
  so changing `[ingest] parser` re-parses. Every parse-cache rewrite runs
  `preserve_cached_metadata` first so user tags/attributes (written by
  `kompend sources tag`/`attr`) survive a forced re-ingest. The chunk cache
  keys on the same fingerprint, so a re-parse also re-chunks.
- Contextual chunk enrichment is opt-in via `search.contextual.enabled`
  (default false). When on, `build_pipeline` constructs a `ContextualEnricher`;
  `_parse.py` runs it after chunking and **before** the chunk-cache write, so the
  cached tree carries `contextual_context`. The chunk-cache key includes the gate
  state, so enabling it invalidates prior un-enriched trees instead of embedding
  stale text. Gate off ⇒ cached output is unchanged.
- **Recoverable-log rule:** a log call that attaches a traceback gates it on
  `kompend.core.verbosity.is_verbose()` (`exc_info=is_verbose()`). Removing a
  traceback is only safe when the one-line message carries the cause: bind the
  exception (`except X as exc`) and append it (`"...: %s", exc`), never leave a
  bare file/key with no reason. `logger.exception(...)` has no gate-able form —
  use `logger.error(..., exc_info=is_verbose())`. The `--verbose` flag itself
  lives in `cli/` (see `cli/AGENTS.md`).
- **Cancel contract** (token: `core.cancel.CancelState`; signal:
  `KompendCancelled`, a `BaseException` so `except Exception` sites cannot
  swallow it): `WikiPipeline.emit()` raises it *before* callback dispatch when
  the token is set — the deliberate exception to "callbacks must not abort the
  pipeline". `run()`/`run_tier()` and `ingest_documents()` stop at tier/file
  boundaries: save tier state, then re-raise. `build_pipeline(config, *,
  cancel=...)` threads the token in; `cli/` owns token creation and the
  double-ESC watcher (see `cli/AGENTS.md`). `_parse.py`'s threaded branch
  shuts its executor down with `cancel_futures=True` so a cancel drops queued
  parse futures instead of chewing through the corpus.
