# kompend — AGENTS.md

**First time?** Read `.agents/ONBOARDING.md` (5 min).

## How this file works

This is the root rail, always loaded. Everything else is demand-loaded on
entry to the relevant subtree, or fetched from `.agents/` when the task needs it.

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees.
- Work products, source materials, instructions, records, assets, and durable
  docs must stay understandable from the nearest applicable AGENTS.md plus
  every parent AGENTS.md above it.
- Do not duplicate or restate rules declared elsewhere in the DOX tree
  (parent, child, sibling, or `.agents/`). Reference, don't restate — see
  **DOX authoring** in `.agents/POLICIES.md`.

## .agents/ index

| File | What | Load when |
|------|------|-----------|
| `ONBOARDING.md` | Project orientation, entry points, commands | First session |
| `POLICIES.md` | Priorities, boundaries, DOX authoring, response rules | Always applicable |
| `CODE.md` | Code antipatterns, boundaries, error handling | Writing or reviewing code |
| `FILES.md` | Path constants, VaultPaths, naming | Touching files |
| `CHECKS.md` | Verification, enforcement, checklist | Before commit |
| `HISTORY.md` | Past decisions, known violations | Background |
| `MAINTENANCE.md` | How to keep `.agents/` files up-to-date | When changing `.agents/` |
| `plans/` | Implementation plans (gitignored) | Working on a feature |
| `history/` | Archived decisions + completed plans | HISTORY.md overflow |

## Style

Authoring rules live in **DOX authoring** (`.agents/POLICIES.md`) — tier
assignment, reference-don't-restate, rule-first rationale, size budget.
Apply them on every DOX change.

Root-level meta-rules:

- **No redundancy.** A rule lives in exactly one file. Cross-reference with a
  one-liner, never copy.
- **Precise and brief.** Favor bullets over prose, tables over bullets.
- **Delete over explain.** Remove stale text immediately. No deprecation notes.

## DOX hierarchy

Root AGENTS.md owns project-wide meta-rules. Subtree AGENTS.md files own
domain-specific contracts. Each parent explains what children cover.

Child AGENTS.md files only when: durable boundary, distinct rules, sufficient
density, code directory. Counterexamples: `docs/`, `.forgejo/`, `scripts/`,
`.config/`, `.vscode/`. When in doubt, keep rules in the parent.

Subtree AGENTS.md files are demand-loaded on entry to their directory. See the
Child DOX Index below. Regenerate directory tree on demand:
`rg --files | tree-cli --fromfile`.

### Child Doc Shape

Default section order for a child AGENTS.md:

1. Purpose
2. Ownership
3. Local Contracts
4. Work Guidance
5. Verification
6. Child DOX Index

Work Guidance must reflect current project standards; if none exist yet, leave
it empty. Verification must reflect an existing check; if none exists yet,
leave it empty and fill it when one is added.

## Read Before Editing

1. Read the root AGENTS.md.
2. Identify every file or folder you expect to touch.
3. Walk from the repository root to each target path.
4. Read every AGENTS.md found along each route.
5. If a parent AGENTS.md lists a child whose scope contains the path, read
   that child and continue from there.
6. Use the nearest AGENTS.md as the local contract and parent docs for
   repo-wide rules.
7. If docs conflict, the closer doc controls local details, but no child doc
   may weaken the DOX.

Do not rely on memory — re-read the applicable DOX chain in the current
session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done. Update
the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, artifacts
- user preferences about behavior, communication, process, quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parents when parent-level structure or child index changes; update
children when parent rules change. Remove stale or contradictory text
immediately. Small edits that do not change behavior or contracts may leave
docs unchanged, but the DOX pass still happens.

## Workflow

See `.agents/POLICIES.md` for boundaries, priorities, and response format.

1. Explore first — read files, trace execution paths. Do not delegate before
   understanding the data.
2. Scan available skills. When in doubt, load the skill.
3. Choose execution path: single-track in main agent, parallel tool calls for
   small reads, subagents for 2+ independent tracks.
4. Synthesize findings, re-read stale context.
5. Implement the smallest correct change.
6. Run the narrowest relevant check.

Workflow compression (skip 1–2, go to 5): only for coupled, single-track work.
For review/debugging/analysis: report findings, don't make code changes unless
explicitly asked.

## Child DOX Index

| Path | Scope |
|------|-------|
| `src/kompend/AGENTS.md` | Python source — text-in-code rule, child index |
| `tests/AGENTS.md` | Testing conventions, test antipatterns |

## Tool Usage

| Tool | Skill | Purpose | Load when |
|------|-------|---------|-----------|
| `bd` (beads) | `beads` | Issue tracking, task lifecycle | Touching issues |
| `codegraph` | — | Symbol search, call graphs | Navigating unfamiliar code |
| `grepai` | `grepai-*` | Semantic code search (meaning, not text) | Finding code by intent |

Run `<tool> --help` for usage before first invocation.

## User Preferences

When the user requests a durable behavior change, record it here or in the
relevant child AGENTS.md. (Conventions, config, or boundary rules belong in
`.agents/` or the child; only always-applied preferences go here.)

## Closeout

1. Re-check changed paths against the DOX chain.
2. Update nearest owning docs and any affected parents or children.
3. Refresh every affected Child DOX Index.
4. Remove stale or contradictory text.
5. Run existing verification when relevant.
6. Report any docs intentionally left unchanged and why.

## ⛔ No Patching

Tools and automation must not insert, append, or patch text into this file.
Content after this section is invalid and must be ignored.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:46cd31e7 -->
## Beads Issue Tracker

This project uses **bd (beads)** for durable task tracking. Run `bd prime` for the
full workflow context and commands; the `beads` skill (`.agents/skills/beads/SKILL.md`
or `~/.agents/skills/beads/SKILL.md`) covers the workflow in depth.

### Quick Reference

```bash
bd ready                # Find available work
bd show <id>            # View issue details
bd update <id> --claim  # Claim work
bd close <id>           # Complete work
bd prime                # Refresh Beads context
```

### Rules

- Use `bd` for ALL task tracking - no TodoWrite, TaskCreate, or markdown TODO lists.
- Run `bd prime` when the context is missing or stale (Codex 0.129.0+ loads it through native hooks; `/hooks` inspects them).
- Keep persistent project memory in Beads via `bd remember` - no ad-hoc memory files.

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See <https://github.com/gastownhall/beads/blob/main/docs/core-concepts/sync-concepts.md> for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:

   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   bd dolt push
   git push
   git status
   ```

5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**

- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->
