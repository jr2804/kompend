# DOX — kompend.memory

<!-- on entry: loaded when agent enters src/kompend/memory -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

kompend memory layer. Two concerns:

- **Query memory** (`store.py`): episodic storage of query/answer/feedback
  triples for semantic retrieval and long-term agent learning.
- **Compile memory** (`lifecycle.py`): confidence scoring, tier promotion,
  supersession, and retention/decay for wiki nodes across compiles.

Both persist to the single vault store (`.kompend/kompend.db`): the query log in
`memory_log`, lifecycle state in `memory_states`. The legacy `memory.json` /
`memory.jsonl` files are no longer written or read (clean-cut store).

## Modules

| Module | Contents |
|--------|----------|
| `store.py` | `MemoryStore`, `MemoryField` — sqlite-vec-backed semantic query memory |
| `lifecycle.py` | `MemoryTier`, `NodeState`, `MemoryStore` (lifecycle), `update_memory_for_compile`, `CompileReport` |
| `__init__.py` | Re-exports `MemoryStore`, `MemoryField`, `update_memory_for_compile` |

## Rules

- Import from the package root for the public API:
  `from kompend.memory import MemoryStore`, or from submodules for specifics:
  `from kompend.memory.lifecycle import update_memory_for_compile`.
- `lifecycle.py` persists to `.kompend/kompend.db` (`memory_states`) via
  `kompend.core.store`; it still depends on `kompend.core.paths` and
  `kompend.core.utils.time`.
- `store.py` depends on `kompend.core.config`, `kompend.core.paths`, `kompend.core.utils.time`,
  `kompend.core.store` (the `memory_log` table), and `kompend.rag.vector_store`.
- **The memory index is brute-force exact.** `vec0` scans the candidates —
  there is no approximate index to add, and none should be: `add_entry()` runs
  once per user query, so an index build would tax every query to save a few
  milliseconds of search. The wiki store is the bulk-built one.
