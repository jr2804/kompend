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

## Project-specific Maintenance — KNOX

### File update triggers

| File | Update when |
|------|-------------|
| `POLICIES.md` | Adding/changing boundary rules, priorities, response format |
| `FILES.md` | Renaming/adding path constants in `paths.py` |
| `CHECKS.md` | Adding/changing verification commands or checklist items |
| `HISTORY.md` | Resolving a known violation or making a notable design decision |
| `ONBOARDING.md` | Project structure or entry points change |
| `plans/` | New feature implementation starts |
| `history/` | HISTORY.md overflow or completed plans archived |

### Cross-reference map

| `.agents/` file | Referenced from |
|-----------------|-----------------|
| `POLICIES.md` | Root `AGENTS.md`, all subtree `AGENTS.md` `on entry` headers |
| `FILES.md` | Root `AGENTS.md`, all subtree `AGENTS.md` `on entry` headers |
| `CHECKS.md` | Root `AGENTS.md`, all subtree `AGENTS.md` `on entry` headers |
| `HISTORY.md` | `src/knox/AGENTS.md` (Known violations section) |
| `ONBOARDING.md` | Root `AGENTS.md` (First time? line) |
| `MAINTENANCE.md` | Root `AGENTS.md` (.agents/ index) |
