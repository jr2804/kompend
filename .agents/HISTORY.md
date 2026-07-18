# AGENTS.md — History

Past design decisions, known issues, and "why we banned X." Optional background
reading; not required for day-to-day work.

## General History

### Pattern: Design Decision Log

Record significant decisions that shaped the codebase. Each entry explains the
context, the decision, and the commit where it landed. This prevents future
contributors from re-litigating settled questions.

### Rules

- **Append-only.** New entries at the top, never delete old ones.
- **Resolved entries.** Move to a "Resolved" subsection rather than removing.
- **Date + commit.** Every entry includes a date and the relevant commit hash.

### Extension

| Who | When | What |
|-----|------|------|
| Any contributor | Making a notable design decision | Add entry at top |
| Project maintainer | Resolving a known violation | Move to Resolved subsection |
| Project maintainer | Deprecating a tool/pattern | Add rationale entry |

---

## Project-specific History — KNOX

### Known text-in-code violations (T1–T10)

Tracked via `bd` (beads). Check current status: `bd list`.

These are LLM prompt strings that live inline in Python modules instead of in
Jinja2 templates under `src/knox/packs/`. They must be migrated before adding
new text-in-code.

| Issue | File | What |
|-------|------|------|
| `knox-ox2` (T1) | `lint/llm_rules.py` | 3 LLM prompt strings for lint rules |
| `knox-8b0` (T2) | `packs/base.py` | 4 default pack prompt strings |
| `knox-1a8` (T3) | `packs/builtins/academic.py` | 3 academic pack prompt strings |
| `knox-65z` (T4) | `packs/builtins/technical.py` | 3 technical pack prompt strings |
| `knox-174` (T5) | `compilation/schema.py` | BUILTIN_KINDS dict with descriptions |
| `knox-dnu` (T6) | `compilation/two_phase.py` | 4 extraction/generation prompt strings |
| `knox-844` (T7) | `compilation/wicer.py` | 3 WiCER loop prompt strings |
| `knox-lf5` (T8) | `verification.py` | 2 claim verification prompt strings |
| `knox-8gj` (T9) | `output/wiki_index.py` | 1 synthesis prompt string |
| `knox-sga` (T10) | `resolution/llm.py` | 1 default entity extraction prompt |

---

### Why text-in-code is banned

Historically, KNOX embedded LLM prompts directly in Python modules. This made
prompts hard to discover, version, and maintain across packs. The rule now is
that all LLM prompts must live in `packs/<name>/templates/` as Jinja2 files.
The T1–T10 violations above are the remaining legacy cases.

---

### Why litellm was removed

Commit `c8055c0`. KNOX originally used litellm for multi-provider LLM dispatch.
After migrating to PydanticAI's native `infer_model()` and the models.dev
catalog, litellm became dead weight — it pulled 40+ transitive dependencies
and added latency. Removed in favor of the unified `build_model()` path in
`src/knox/llm/`.

---

### Release versioning

Releases follow `YYYY.M.D` / `YYYY.M.D.N` calver format (no zero-padding, no
`v` prefix). See `.forgejo/workflows/release.yml`.
