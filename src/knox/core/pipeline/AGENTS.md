# DOX — knox.core.pipeline

## Purpose

5-tier pipeline orchestration: index → chunk → resolve → compile → output.
Contains the orchestrator, pipeline configuration dataclass, tier state
persistence, and standalone ingestion.

## Modules

| Module | Contents |
|--------|----------|
| `orchestrator.py` | `WikiPipeline`, `build_pipeline`, `build_llm_client`, `ProgressCallback` |
| `config.py` | `Tier`, `Compiler`, `ALL_TIERS`, `TIER_LEVEL`, `TIER_DESCRIPTIONS`, `PipelineConfig` |
| `state.py` | `TierStateStore`, `TierState`, `SourceState`, `CompileStatus`, `classify_compile_status` |
| `ingest.py` | `ingest_documents`, `parse_cache_name`, `parse_cache_glob`, `IngestResult` |

## Rules

- Import from submodules directly: `from knox.core.pipeline.orchestrator import build_pipeline`,
  `from knox.core.pipeline.config import PipelineConfig`.
- `config.py` is the public config shape — both CLI and orchestrator import it.
- `orchestrator.py` imports from nearly every other core submodule; this is
  expected for an orchestrator.
