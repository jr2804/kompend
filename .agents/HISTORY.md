# AGENTS.md — History

Past design decisions, known issues, and "why we banned X." Optional background
reading; not required for day-to-day work.

## General History

### Pattern: Design Decision Log

Record significant decisions that shaped the codebase. Each entry explains the
context, the decision, and where it landed. This prevents future contributors
from re-litigating settled questions.

### Rules

- **Newest first**, **one entry per decision**, 2–6 lines: context, decision,
  commit. Detail belongs in `.agents/history/` — link it.
- **Budgeted like every `.agents/` file (≤120 lines).** Over budget → archive the
  oldest entries to `.agents/history/`, leaving a pointer line.
- Move resolved entries to a "Resolved" subsection rather than deleting them.

### Extension

| Who | When | What |
|-----|------|------|
| Any contributor | Making a notable design decision | Add entry at top |
| Project maintainer | Resolving a known violation | Move to Resolved subsection |
| Project maintainer | Deprecating a tool/pattern | Add rationale entry |

---

## Project-specific History — kompend

### Archived plans — full detail in `.agents/history/`

| Archived | Slug | What landed |
| -------- | ---- | ----------- |
| 2026-09-16 | `context-window-provider-overrides` | `[providers.<name>].context_window` now beats auto-discovery (step 1 of `discover_context_window`); `kompend doctor` stops reporting override-only blocks of *known* providers as `missing base_url` while still erroring on unknown ids |
| 2026-09-24 | `kompend-0iv2-audit` | Read-only audit resolving a tracker-vs-code contradiction from the code; the reusable part is the method — dynamic-access sweep plus `git log -S "<Class>"` |
| 2026-09-24 | `kompend-storage-migration` | `.kompend/` consolidated into one SQLite file (`core/store/`); clean cut, no migration code — `61147c5f`, `f6aee4c0`, `8aaf2eca` |
| 2026-08-22 | `markdown-only-vault` | The wiki is represented by its generated markdown alone; `.kompend/` is purely derived — `5d493119`, `51ea1111` |
| 2026-08-16 | `source-export-md`, `category-extraction`, `rumdl-format-wiring` | `kompend sources export` from the parse cache; configurable extraction categories; rumdl wiring for every markdown writer |
| 2026-08-15 | `env-any-refactor`, `rlm-integration` | Typed config access replacing `Any`; the optional RLM code-exploration capability |
| 2026-08-14 | `migration-plan`, `refactoring-plan` | `summarization-pydantic-ai` context management (`edb4d49`); super-module splits into `stages/`, `engine/`, `tools/`, `flows/`, `pages/` |
| 2026-08-14 | `lint-remediation-collapse` | Three overlapping lint tiers collapsed to LLM-generated concept pages; `image-embed` became a `run --fix` structural rule, `RemediationAction.tier` removed |

### Archived decisions — full detail in `.agents/history/`

| Archived | Slug | What landed |
| -------- | ---- | ----------- |
| 2026-10-27 | `notebook-vault-stages` | The vault notebook reaches every stage that keeps one — summarisation, verification, remediation, distillation — through `build_notebook_store`; the harness namespaces each notebook by agent name, so the stages share a directory and never a file |
| 2026-10-27 | `notebook-cache-decoupling` | The agent notebook left the summary cache key: the notebook is agent-written prompt context, so it cannot invalidate cached summaries (the old `MEMORY.md` fingerprint never fired and would have thrashed the cache) |
| 2026-10-27 | `agent-memory-default` | `compile.llm.agent_memory` defaults to `true` and `compile.llm.memory_store` to `"vault"` — the notebook is on for the four stages that handle many outputs in one session; `agent_memory = false` remains the escape hatch for small models |
| 2026-10-27 | `harness-transcripts` | Generation checkpoints (`compilation/_checkpoints.py`) replaced by harness `StepPersistence` transcripts in `SqliteStepStore` (`.kompend/steps.db`); `step_recovery.py` derives the deterministic keys and resumes/skips segments from the transcript; ConversationSearch + SystemReminders wired onto the agentic generation and summarizer agents |
| 2026-10-26 | `vault-agent-memory` | `compile.llm.memory_store = "vault"` persists the agent notebook at `.kompend/agent-memory/` (first-party `DirectoryMemoryStore` - the harness `FileStore` is sandbox-first with a POSIX-only local backend) |
| 2026-10-23 | `rationale-archive` | text-in-code ban, litellm removal, lazy imports (SPEC-0001), `compile.embedding.dim` removal, `_ENV_LIST_VARS` schema derivation, fixed CLI violations table, `functools.cache` over `global` |
| 2026-10-23 | `dependency-upgrades` | pydantic-ai 2.60.0 / harness 0.60.0 bump, anthropic pin dropped, `[tool.uv]` overrides for openai and typer, sentence-transformers dropped (pydantic-ai-slim, remote embedding providers only) |

### Release versioning

Calver `YYYY.M.D` / `YYYY.M.D.N` — no zero-padding, no `v` prefix. See
`.forgejo/workflows/release.yml`.
