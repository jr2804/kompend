# DOX — kompend.core.resolution

<!-- on entry: loaded when agent enters src/kompend/core/resolution -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Programmatic Tier 3 resolution: entity extraction/deduplication and
cross-reference consistency checking — no LLM needed.

## Modules

| Module | Contents |
|--------|----------|
| `entity_registry.py` | `EntityRegistry`, `EntityRecord`, `EntityLocation` |
| `xref_checker.py` | `CrossReferenceChecker` |

## Rules

- Import from submodules: `from kompend.core.resolution.entity_registry import EntityRegistry`.
- Both modules depend on `kompend.core.models` for `ChunkTree`, `StructuredData`,
  and cross-reference result types.
- `xref_checker.py` depends on `entity_registry.py` — they work together.
