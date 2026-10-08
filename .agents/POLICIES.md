# AGENTS.md — POLICIES

Always applicable. Priorities, boundaries, DOX authoring, response format.

## Priorities

Correctness · Evidence · Safety · Minimal changes · Consistency · Performance

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
- NEVER create temp files outside `tmp/` (gitignored) — never `_tmp_*.py` at the root.

## Change constraints

- Minimal, surgical edits. Preserve existing style. No unrelated refactoring.
- When changing tests, update only directly affected tests.

## Code quality

Universal defaults. KNOX code policies — antipatterns, boundaries, error
handling — live in `CODE.md`; project-specific standards in child AGENTS.md.

- **SRP** — one reason to change per module/function. If a name needs "and",
  split it. Remove dead branches, unreachable conditions, and unused params or
  imports before finishing.
- **DRY** — check for an existing helper/constant before adding one; extract on
  the _third_ occurrence (Rule of Three), not earlier.
- Prefer a little duplication over the wrong premature abstraction; add no new
  code-quality tooling, patterns, or value-object rules by default.

## Completion checklist

Change solves the stated problem · relevant validation ran (or gaps explicitly
stated, see `CHECKS.md`) · no unintended side effects · no secrets added

## DOX authoring (keep AGENTS.md lean)

AGENTS.md and `.agents/` files share limited context. Bloat and duplication are
the failure mode — they waste context on every subtree entry and drift apart as
rules change.

### Where a rule lives

Ask: **what is the highest (most general) tier that fully applies?** Put the rule
there and nowhere else.

- All code in the project → `.agents/POLICIES.md` (universal) or `CODE.md` (code
  policies). This is the default; when unsure, here.
- Only inside one subtree (framework-specific template rules) → that subtree's
  `AGENTS.md`.
- Only navigation → the parent's Child DOX Index, nothing else.

### Reference, don't restate

A rule appears in **exactly one file**; every other file touching it uses a
**pointer line** — `<topic> — see <file>`. Never copy the rule, its rationale, or
its example into a second file.

### Rule first, rationale second, never third

A DOX line is the rule. If a rationale is costly to rediscover, add one short
clause. Never multi-paragraph explanations or more than one ❌/✅ pair per rule —
those belong in a skill or a `docs/` page, linked once.

### Content rules

- **No tree views** — generate on demand: `rg --files | tree-cli --fromfile`.
- **No history** — git log has it; record only decisions costly to rediscover,
  in `HISTORY.md` with commit refs.
- **No TODO lists or trackers** — use beads (`bd`).
- **Long-form explanation** belongs in `docs/` or a skill: link it, don't inline.

### Size budget

- `.agents/` files: ≤ **120 lines** (always-injected context).
- Subtree `AGENTS.md`: ≤ **250 lines** (loaded cumulatively with every parent on
  entry — deep-path cost adds up).

Over budget means the doc restates instead of pointing, or hoards rationale. Cut
first; split the subtree only as a last resort.

## Response format

Concise and specific. No filler, intros, or restated requirements. Answer direct
questions directly. For review/debugging/analysis: findings with references,
conclusion, approach, and caveats.

## Project-specific Policies — KNOX

### Implementation plans

- Plans go in `.agents/plans/<feature>/` — gitignored, never committed. Mutable
  working artifacts, not shipped docs. Prefix filenames with phase numbers so
  file order matches implementation order; carry `Created:` and `Status:` lines.
  Shipped docs (user-facing guides, API references) go in `docs/` and ARE
  committed.

### Documentation and archiving

- `HISTORY.md` over budget → archive older entries to `.agents/history/` with a
  one-line pointer left behind. `.agents/history/` IS version-controlled; it
  holds the durable decision records.
- Completed plans move to `.agents/history/` as `YYYY-MM-DD-<slug>.md`
  (completion date), with references updated. See `MAINTENANCE.md`.

### User Preferences

- Concise, direct answers with no filler or intros
- Use `uv run` for all Python execution, never direct `python` calls
- English-only for code, docs, and commits; conventional commits (`feat:`, `fix:`,
  `chore:`, `docs:`)
