# DOX — knox.core.pipeline

<!-- on entry: loaded when agent enters src/knox/core/pipeline -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

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
| `ingest.py`           | `ingest_documents`, `parse_cache_name`, `parse_cache_glob`, `preserve_cached_metadata`, `parse_cache_is_current`, `PARSE_CACHE_PARSER_KEY`, `IngestResult` |

## Rules

- Import from submodules directly: `from knox.core.pipeline.stages._pipeline import WikiPipeline`,
  `from knox.core.pipeline.config import PipelineConfig`.
- Public re-exports in `__init__.py` preserve backward compatibility —
  `from knox.core.pipeline.orchestrator import WikiPipeline` still works.
- `config.py` is the public config shape — both CLI and orchestrator import it.
- `_pipeline.py` imports from nearly every other core submodule; this is
  expected for an orchestrator.
- Each stage module keeps its responsibility isolated — stages communicate
  through the `WikiPipeline` state and shared data models.
- The parse cache is parser-aware: entries carry `PARSE_CACHE_PARSER_KEY`
  (from `ParserRouter.fingerprint`) and a mismatched fingerprint is a miss,
  so changing `[ingest] parser` re-parses. Every parse-cache rewrite runs
  `preserve_cached_metadata` first so user tags/attributes (written by
  `knox sources tag`/`attr`) survive a forced re-ingest. The chunk cache
  keys on the same fingerprint, so a re-parse also re-chunks.
