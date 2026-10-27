# The vault notebook reaches every notebook stage (2026-10-27)

## Why

`compile.llm.memory_store = "vault"` shipped with one consumer: the
summariser. The three other stages that keep a notebook — verification,
remediation, distillation — had no store parameter, so their notes died with
the process even though the config asked for the vault.

The original worry was a collision: every stage writing
`.kompend/agent-memory/MEMORY.md`. It does not happen — the harness scopes
every notebook path by agent name (`scope = f'{namespace}/{agent_name}'`), so
each stage owns `<agent_memory_dir>/<agent_name>/MEMORY.md`. The stages share
a directory, never a file.

## What landed

- `build_notebook_store(llm, memory_dir)` (`src/kompend/agents/vault_memory.py`)
  is the one place that decides: the notebook is on, the store is `vault`, and
  a directory is given: a `DirectoryMemoryStore`; otherwise `None` (the
  harness then uses its in-process store).
- The four notebook stages take `memory_dir` and use that helper: the
  summariser (`summarizer_agentic.py`), verification (`verification.py`),
  remediation (`remediate.py`) and distillation (`distill_agentic.py`).
- The callers pass the vault's agent-memory directory: the pipeline factory
  (`core/pipeline/stages/_factory.py`), the verification path
  (`cli/_post_compile.py`), the remediation paths (`cli/cmds/quality/lint.py`,
  `lint/flow.py`, `mcp/tools/quality.py`) and the distillation paths
  (`cli/_distill_helpers.py` + its callers, `mcp/tools/knowledge.py`).

## Consequence

A stage keeps its notebook across runs. That is the point of the vault
backend: a verification pass remembers what it already checked.

## Known gap

The stages that build a fresh agent per unit (generation, write, query,
context-chunk, overview) keep no notebook at all — a notebook there would be
empty every unit. They no longer pass a `memory_agent_name`, so nothing
suggests otherwise (bead `knox-u6qg`).
