# AGENTS.md — Maintenance

How to keep `.agents/` files current.

## General Maintenance

### Process

1. When a rule changes, update the file that owns it — never another file.
2. When adding a new `.agents/` file, add it to the `.agents/ index` table
   in root `AGENTS.md`.
3. After any `.agents/` change, check the AGENTS.md index for stale entries.

### Principles

- **Single source of truth.** Each rule lives in exactly one file.
  Cross-reference with a one-liner, never duplicate.
- **AGENTS.md is an index.** It points to `.agents/` files; it does not contain
  the rules themselves.
- **No tree figures.** ASCII directory trees rot immediately. Regenerate on
  demand: `rg --files | tree-cli --fromfile`.

### Extension

| Who | When | What |
|-----|------|------|
| Project maintainer | Adding a new `.agents/` file | Add to AGENTS.md index |
| Project maintainer | Renaming a `.agents/` file | Update AGENTS.md + cross-references |
| Any contributor | Finding stale references | Fix or report |

---

## Project-specific Maintenance — kompend

### File update triggers

| File | Update when |
|------|-------------|
| `POLICIES.md` | Adding/changing priorities, boundaries, DOX authoring, response format |
| `CODE.md` | Adding/changing a code antipattern, boundary, or the error-handling policy |
| `FILES.md` | Renaming/adding path constants in `paths.py` |
| `CHECKS.md` | Adding/changing verification commands or checklist items |
| `HISTORY.md` | Resolving a known violation or making a notable design decision |
| `ONBOARDING.md` | Project structure or entry points change |
| `plans/` | New feature implementation starts |
| `history/` | HISTORY.md overflow or completed plans archived |

### Plan naming and dating

- Active plans in `plans/` keep their working name (no date prefix) but must
  carry date info in the document (e.g. a `Created: YYYY-MM-DD` line under
  the title; update a `Status:` line as work progresses).
- On archiving into `history/`, rename to `YYYY-MM-DD-<plan-slug>.md` where
  the date is the **completion date** (e.g. `2026-08-14-migration-plan.md`).
  Update any references to the old filename (HISTORY.md entries, links).

### Cross-reference map

| `.agents/` file | Referenced from |
|-----------------|-----------------|
| `POLICIES.md` | Root `AGENTS.md`, all subtree `AGENTS.md` `on entry` headers |
| `CODE.md` | Root `AGENTS.md`, subtree `AGENTS.md` `on entry` headers, `tests/AGENTS.md` |
| `FILES.md` | Root `AGENTS.md`, all subtree `AGENTS.md` `on entry` headers |
| `CHECKS.md` | Root `AGENTS.md`, all subtree `AGENTS.md` `on entry` headers |
| `HISTORY.md` | Root `AGENTS.md` (.agents/ index) |
| `ONBOARDING.md` | Root `AGENTS.md` (First time? line) |
| `MAINTENANCE.md` | Root `AGENTS.md` (.agents/ index) |
