# DOX — knox.core.resolution

<!-- on entry: loaded when agent enters src/knox/core/resolution -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Programmatic Tier 3 resolution: entity extraction/deduplication and
cross-reference consistency checking — no LLM needed.

## Modules

| Module | Contents |
|--------|----------|
| `entity_registry.py` | `EntityRegistry`, `EntityRecord`, `EntityLocation` |
| `xref_checker.py` | `CrossReferenceChecker` |

## Rules

- Import from submodules: `from knox.core.resolution.entity_registry import EntityRegistry`.
- Both modules depend on `knox.core.models` for `ChunkTree`, `StructuredData`,
  and cross-reference result types.
- `xref_checker.py` depends on `entity_registry.py` — they work together.
