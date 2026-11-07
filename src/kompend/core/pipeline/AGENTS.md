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
| `stages/_parse.py`    | Tier 1 parse + chunk orchestration; heavy parses run under `HeavyParseGuard` (`preprocessing/host_memory.py`) |
| `stages/_index.py`    | Tier 0 source indexing                                                                   |
| `stages/_resolve.py`  | Tier 3 resolution orchestration                                                          |
| `stages/_compile.py`  | Tier 4 compilation + Tier 5 output orchestration                                         |
| `config.py`           | re-exports `Tier`, `Compiler`, `TierInfo` from `core.config`; `PipelineConfig`            |
| `state.py`            | `TierStateStore`, `TierState`, `SourceState`, `CompileStatus`, `classify_compile_status` |
| `references.py`       | `ReferenceStore` — JSON-backed bibliographic reference persistence                       |
| `ingest.py`           | `ingest_documents`, `parse_cache_name`, `preserve_cached_metadata`, `parse_cache_is_current`, `PARSE_CACHE_PARSER_KEY`, `parse_target_path`, `IngestResult` |
| `parallelism.py`      | `gil_enabled`, `describe_parallelism` — interpreter parallelism facts for `compile.workers` |
| `bounded_parse.py`    | `DeadlineWorker`, `ParseDeadlineExceeded` — per-file wall-clock budget, enforced by a reusable worker process (`ingest.parse_timeout_s`; `0` parses in-process) |
| `reasons.py`          | `reason_code_for` — maps a raised/worker-flattened exception to its `ReasonCode` |
| `attachments.py`, `assets.py` | Attachment ingestion (`ingest_attachment`), asset store — item-path modules |
| `export.py`           | Tier-1 cache → markdown export (`export_source`, `render_document_markdown`); text elements pass through `core.text.pipes.escape_prose_pipes` (generation contract) |

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
- The parse cache is content-addressed: an entry is reused when the source
  bytes hash to the same key **and** the entry was written by the current
  backend identity (`parser.backend_identity()`, stored under
  `PARSE_CACHE_PARSER_KEY`, checked by `parse_cache_is_current`) — a changed
  backend set or `[ingest]` restriction re-parses, a parsecraft version bump
  does not. Entries from before the identity existed store an empty key and
  re-parse once (self-healing). `compile.reparse` (CLI `--reparse`) bypasses the
  parse and chunk caches — sources are re-extracted and re-chunked, cached
  LLM results stay; `compile.no_cache` bypasses every cache, LLM results
  included. Every parse-cache rewrite runs `preserve_cached_metadata` first
  so user tags/attributes (written by `kompend sources tag`/`attr`) survive a
  forced re-ingest.
- **A source is identified by its name plus a content-hash prefix** —
  `kompend.core.identity.source_key(stem, content_hash)` → `<stem>-<hash8>` —
  never by its path. The parse cache, the chunk cache and `tier_state` all use
  that one rule, so grouping sources into folders or moving them under
  `sources/` keeps every cache entry and every state row, while two files
  sharing a name but differing in content stay two sources. A cache hit is
  rebound to the path the current run looked it up for (`_load_cache`,
  `_try_load_chunk_cache`) so no consumer resurrects a path that no longer
  exists; `tier_state.source_file` is only the last path the source was seen
  at.
- The parse tier and the standalone ingest share one decision point for
  *which bytes a parser receives*: `parse_target_path` (in `ingest.py`) hands
  a known text format over as UTF-8 and every other source byte-wise. One
  refusal rides the text branch: bytes that sniff as binary behind a
  text-classified suffix are a mislabelled file (a PDF named `.md`) and
  raise `BinarySourceError` via `assert_text_source`
  (`preprocessing/parsers/base.py`) instead of normalizing U+FFFD soup into
  a successful garbage parse (knox-rtmn). Everything else passes — the
  parser already picked a backend, and the format table classifies bytes,
  it does not gate parsing. The UTF-8 sibling keeps the source suffix
  (`<hash><suffix>`), so the backend still classifies it, and every caller
  of `parse_target_path` runs `restore_source_path` straight after parsing:
  documents and elements carry the original path, the sibling never reaches
  citations or the parse cache.
- Contextual chunk enrichment is opt-in via `search.contextual.enabled`
  (default false). When on, `build_pipeline` constructs a `ContextualEnricher`;
  `_parse.py` runs it after chunking and **before** the chunk-cache write, so the
  cached tree carries `contextual_context`. The chunk-cache key includes the gate
  state, so enabling it invalidates prior un-enriched trees instead of embedding
  stale text. Gate off ⇒ cached output is unchanged.
- The resolve tier hands the caption prompt its surroundings: for every
  image chunk it assembles the section (`chunk.heading_context`), the
  neighbouring text of the same tree and the LLM-written
  `contextual_context`, and passes them on in
  `chunk.metadata["vlm_context"]` (`MetadataField.VLM_CONTEXT`). The
  resolve cache key carries that context's hash and the identity of the
  caption templates (`vlm/caption.md.j2`, `vlm/context.md.j2`), so a
  changed figure or a changed prompt cannot replay a stale caption. The
  context is a per-run value: the chunk cache is written before the
  resolve tier, so it never enters it.
- Heavy parses are budgeted, not just threaded: a source that may load the
  heavy conversion engine (non-text `FileExtension` — PDF/office/image)
  enters `HeavyParseGuard` (`preprocessing/host_memory.py`): at most
  `ingest.max_concurrent_heavy_parses` run at once and a heavy parse needs
  `ingest.parse_min_free_commit_mb` of free commit charge (measured peak of
  one docling conversion: ~8 GB, independent of page count — knox-b9p2). A
  refused source is skipped loudly per-file; the run continues. `0` disables
  the headroom check, the cap stays. The docling engine's ONNX thread count
  (`ingest.parsecraft_docling_threads` → `DOCLING_RS_PDF_THREADS`, set at
  parser construction, before the first conversion) is the engine-side
  lever; the default is `min(4, cores)` — the measured knee (7.4 s / 1830 MB
  vs 7.1 s / 2790 MB at the engine default and 16.3 s / 1533 MB at one
  thread).
- **Recoverable-log rule:** a log call that attaches a traceback gates it on
  `kompend.core.verbosity.is_verbose()` (`exc_info=is_verbose()`). The switch is
  a `ContextVar`, so a worker thread only sees it when the submission carries
  the context (`_context_bound_call` in `stages/_parse.py` wraps each pool
  submission in `copy_context().run`). Removing a
  traceback is only safe when the one-line message carries the cause: bind the
  exception (`except X as exc`) and append it (`"...: %s", exc`), never leave a
  bare file/key with no reason. `logger.exception(...)` has no gate-able form —
  use `logger.error(..., exc_info=is_verbose())`. The `--verbose` flag itself
  lives in `cli/` (see `cli/AGENTS.md`).
- **Reason-code rule:** every item-path `logger.error`/`logger.warning` carries
  `extra={"reason_code": ReasonCode.X}` — the enum from `kompend.core.types`, never
  a bare string. Sites that catch a concrete exception pass the member directly;
  broad `except Exception` sites classify with `reason_code_for(exc)`
  (`reasons.py`). The observability ledger, `kompend explain` and doctor count
  on this field; `tests/unit/test_reason_codes.py` pins the emission sites.
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
