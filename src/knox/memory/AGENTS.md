# DOX — knox.memory

<!-- on entry: loaded when agent enters src/knox/memory -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

KNOX memory layer. Two concerns:

- **Query memory** (`store.py`): episodic storage of query/answer/feedback
  triples for semantic retrieval and long-term agent learning.
- **Compile memory** (`lifecycle.py`): confidence scoring, tier promotion,
  supersession, and retention/decay for wiki nodes across compiles.

Both persist to the single vault store (`.knox/knox.db`): the query log in
`memory_log`, lifecycle state in `memory_states`. The legacy `memory.json` /
`memory.jsonl` files are no longer written or read (clean-cut store).

## Modules

| Module | Contents |
|--------|----------|
| `store.py` | `MemoryStore`, `MemoryField` — zvec-backed semantic query memory |
| `lifecycle.py` | `MemoryTier`, `NodeState`, `MemoryStore` (lifecycle), `update_memory_for_compile`, `CompileReport` |
| `__init__.py` | Re-exports `MemoryStore`, `MemoryField`, `update_memory_for_compile` |

## Rules

- Import from the package root for the public API:
  `from knox.memory import MemoryStore`, or from submodules for specifics:
  `from knox.memory.lifecycle import update_memory_for_compile`.
- `lifecycle.py` persists to `.knox/knox.db` (`memory_states`) via
  `knox.core.store`; it still depends on `knox.core.paths` and
  `knox.core.utils.time`.
- `store.py` depends on `knox.core.config`, `knox.core.paths`, `knox.core.utils.time`,
  `knox.core.store` (the `memory_log` table), and `knox.rag.zvec_store`.
- `store.py` keeps zvec's default **flat** index for
  `MemoryField.QUERY_EMBEDDING` — it is built inline on insert, always
  complete, with zero insert overhead. Do NOT add `HnswIndexParam` or per-insert
  `optimize()` here: `add_entry()` runs once per user query, so HNSW's ~40 ms
  `optimize()` cost would make every query slower than the ~8 ms flat search it
  saves. HNSW belongs on the bulk-built wiki store, not this one.
