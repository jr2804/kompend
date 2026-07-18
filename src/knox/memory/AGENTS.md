# DOX — knox.memory

<!-- on entry: loaded when agent enters src/knox/memory -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

KNOX memory layer. Two concerns:

- **Query memory** (`store.py`): episodic storage of query/answer/feedback
  triples for semantic retrieval and long-term agent learning.
- **Compile memory** (`lifecycle.py`): confidence scoring, tier promotion,
  supersession, and retention/decay for wiki nodes across compiles.

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
- `lifecycle.py` depends on `knox.core.paths` and `knox.core.utils.time`.
- `store.py` depends on `knox.core.config`, `knox.core.paths`, `knox.core.utils.time`,
  and `knox.rag.zvec_store`.
