# The agent notebook is not a cache input (2026-10-27)

## Why the coupling was dropped

The vault backend (2026-10-26, bead `knox-7h48`) fingerprinted `MEMORY.md`
into the summary cache deps, so editing the notebook invalidated cached
summaries like a model switch. The intent was sound — the harness re-injects
the notebook snapshot before every model request, so the notebook is prompt
input — but the implementation could not deliver it:

- **It never fired.** Kompend fingerprinted `.kompend/agent-memory/MEMORY.md`
  (top level), while the harness scopes every notebook path by agent name
  (`scope = f'{namespace}/{agent_name}'`, `path = f'{scope}/{name}'`). The
  summariser's agent name is `summarizer`, so the real file is
  `.kompend/agent-memory/summarizer/MEMORY.md`. `_memory_fingerprint` returned
  `None` in every real run and `deps["memory"]` was never set. The unit test
  passed only because it hand-planted the top-level file.
- **Pointing at the right path would thrash the cache.** The notebook is
  agent-written — the harness guidance tells the model to store facts
  proactively — while the fingerprint was computed once at pipeline
  construction, before the run. The agent rewrites the notebook mid-run, so
  the file never matches the fingerprint the run cached under, and the next
  run's cache key changes: every summary re-derives, forever. It could not be
  exact even within one run, because the injected snapshot changes between
  requests.

The notebook is prompt context the pipeline itself mutates — not a stable
cache input. The coupling was dropped instead of repaired.

## What landed

- `NodeSummarizer` lost the `memory_fingerprint` parameter and the `memory`
  cache dep (`src/kompend/chunking/summarizer.py`); the summary cache key is
  content + heading chain + model again.
- The pipeline factory lost `_memory_fingerprint` and the `MEMORY_INDEX_FILE`
  import (`src/kompend/core/pipeline/stages/_factory.py`);
  `MEMORY_INDEX_FILE` is gone from `src/kompend/core/paths.py` — the harness
  owns the notebook file name.
- `kompend.toml.example`, `docs/guides/llm-config.md`,
  `src/kompend/core/config/_configs.py`, `src/kompend/agents/AGENTS.md` now
  say what is true: a `MEMORY.md` index per agent, and editing it leaves
  cached summaries as they are.

## Consequence

Editing the notebook no longer invalidates cached summaries. That is the
point: the notebook is agent-written, so an edit means the agent learned
something new, not that the inputs changed. Use `compile.force` (or a model
switch) to re-derive summaries.

## Known gap (closed 2026-10-27)

The vault backend reached the summariser only; verification, remediation and
distillation kept the in-process store. Closed by `notebook-vault-stages`
(bead `knox-lwvw`) — the harness namespaces notebook paths by agent name, so
wiring the other three was plumbing, not a redesign.
