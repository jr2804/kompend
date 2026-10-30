# AGENTS.md — POLICIES

Always applicable. Priorities: correctness · evidence · safety · minimal changes ·
consistency · performance. Boundaries, DOX authoring, response format.

## Boundaries

- NEVER fabricate paths, commits, APIs, config keys, env vars, test results, or
  benchmark numbers. If you don't know, say so.
- NEVER guess at command names, flags, or paths — read source or run `--help`.
- NEVER add secrets, API keys, or tokens to files; use env vars.
- NEVER run destructive commands (`rm -rf`, `git reset --hard`,
  `git push --force`) without explicit confirmation.
- NEVER delete or move files without explicit instruction.
- NEVER modify `.gitignore`, `pyproject.toml`, `ruff.toml`, or CI config, and add
  no new dependency, without explicit instruction.
- NEVER install with `pip install` or `uv pip install`: uv alone manages the venv
  (`uv add`, `uv lock`, `uv sync`); pip bypasses `uv.lock` — untracked, never relied on.
- NEVER create temp files outside `tmp/` (gitignored) — never `_tmp_*.py` at the root.
- NEVER commit, push, or sync a tracker unless the user asked for it.

## Change constraints

- Minimal, surgical edits; preserve existing style; no unrelated refactoring.
  When changing tests, update only directly affected tests.

## Code quality

Antipatterns, boundaries and error handling live in `CODE.md`; project-specific
standards in child AGENTS.md. Universal defaults:

- **SRP** — one reason to change per module/function. If a name needs "and",
  split it; remove dead branches and unused params/imports before finishing.
- **DRY** — check for an existing helper/constant before adding one; extract on
  the _third_ occurrence (Rule of Three), not earlier.
- A little duplication beats a premature abstraction; add no new quality tooling by default.

## Completion checklist & verification

Change solves the stated problem · validation ran (or gaps stated) · no unintended
side effects · no secrets added. Commands, enforcement scripts, and the new-code
checklist live in `CHECKS.md` — run the narrowest relevant one before finishing.

## DOX authoring (keep AGENTS.md lean)

AGENTS.md and `.agents/` files share limited context; bloat and duplication are
the failure mode.

### Where a rule lives

Put a rule in the **highest (most general) tier that fully applies**, nowhere else.

- All code in the project → `.agents/POLICIES.md`, or `CODE.md` for code policies
  (the default; when unsure, here).
- Only inside one subtree → that subtree's `AGENTS.md`.
- Only navigation → the parent's Child DOX Index.

### Reference, don't restate

A rule appears in **exactly one file**; every other file uses a pointer line —
`<topic> — see <file>`. Never copy the rule, rationale, or example.

### Rule first, rationale second, never third

A DOX line is the rule; add one short clause only if the rationale is costly to
rediscover. Multi-paragraph explanations and ❌/✅ pairs belong in a skill or a
`docs/` page, linked once.

### Content rules

- **No tree views** — generate on demand: `rg --files | tree-cli --fromfile`.
- **No history in AGENTS.md files** — git owns routine history; record only
  costly-to-rediscover decisions in `HISTORY.md`, or `.agents/history/`.
- **No TODO lists or trackers** — use beads (`bd`).
- **Long-form explanation** belongs in `docs/` or a skill: link it, don't inline.

### State file subdirectories

A state file stays a lean index and its matching subdirectory carries the detail
— created only when the file approaches its budget, with every file in it listed
from the owning state file.

| State file | Optional subdirectory |
| ---------- | --------------------- |
| `FILES.md` | `.agents/files/` |
| `POLICIES.md` | `.agents/policies/` |
| `TOOLS.md` | `.agents/tools/` |
| `MAINTENANCE.md` | `.agents/maintenance/` |

`ONBOARDING.md` and `HISTORY.md` have none — history growth goes to
`.agents/history/`.

### Size budget

`.agents/` files ≤ **120 lines** (always injected); subtree `AGENTS.md` ≤ **250**
(loaded cumulatively from every parent on entry). Over budget means restating
instead of pointing — cut first, split a subtree only as a last resort.

## Response format

Concise and specific. No filler, intros, or restated requirements. For
review/debugging/analysis: findings with references, conclusion, caveats.

## Implementation plans

- Plans go in `.agents/plans/<YYYY-MM-DD>-<feature-name>/` — gitignored, phase
  files numbered (`01-design.md`). Shipped docs go in `docs/` and ARE committed.
- Archive on completion, not at a later pass: consolidate into
  `.agents/history/<YYYY-MM-DD>-<plan-name>.md`, delete the plan directory, and
  reference the archive from `HISTORY.md` (procedure: `MAINTENANCE.md`).

## Project-specific policies — kompend

### User preferences

- Concise, direct answers with no filler or intros.
- English-only for code, docs, and commits; conventional commits (`feat:`, `fix:`,
  `chore:`, `docs:`).
