# Refactoring a growing codebase — inputs for a skill

Status: raw material for a future `/refactor` skill, collected 2026-11-06 from
one long session in this repository. Everything above the appendix is
project-agnostic on purpose; the appendix holds the evidence that produced it
and is there for calibration, not for copying.

## 0. What the skill must encode

| Question | Answer |
|---|---|
| Trigger | an explicit request ("refactor/split/clean up X") or a measured signal from §1 that crosses the repo's own median |
| Input | one folder, one concern — never "the codebase" |
| Output | commits (one per mechanical step), a before/after inventory, updated docs |
| Stop condition | the §1 ranking stops paying (see §5.12) |
| Forbidden | mixing a move with a behaviour change; weakening tests to pass; refactoring what cannot be run |

Not a trigger: "the code looks ugly", a pending feature (build it, then
refactor), a bug (fix it first, separately).

## 1. Status — what "needs refactoring" actually looks like

Measure, never opine. Compare against the repo's own median, not an absolute
truth: a 300-line module is unremarkable in a repo whose median is 280 and a
smell in one whose median is 40.

| Signal | Measure | Red flag | Why it matters |
|---|---|---|---|
| File size | lines per leaf module | far above the repo median, or a leaf >300 | the file is a container, not a concept |
| Function size | statements per function | >60, or a suppression (`# noqa: PLR0915`, `C901`) | nobody can hold it in working memory |
| Suppression density | suppressions per file | more than a couple | each one is a rule the code cannot meet |
| Naming | `utils`, `helpers`, `_shared`, `common`, `misc`, `_anything` | any | the name says "not my job", so nobody finds anything |
| Reasons to change | list the module's unrelated "and also" claims | >1 | the definition of a missing boundary |
| Fan-in | importing files per module | top decile of the repo | blast radius of every edit |
| Churn | commits touching the file over 6 months | top decile | where the pain actually is |
| Test mirror | test file size vs module size | test ≥ module, or a 1:1 giant | the module has no seam |
| Duplicate vocabulary | hand-written lists that mirror an enum/schema | any | drift by construction |
| Doc drift | responsibility tables vs the code | mismatch | the map lies, so reviews miss things |

```sh
# size, descending (swap the glob for the language)
rg --files -g '*.py' | xargs wc -l | sort -rn | head -20
# suppression density
rg -c "noqa|type: ignore|eslint-disable|@ts-ignore"
# fan-in per module
rg -o "from app\.(\w+)" -r '$1' --no-filename | sort | uniq -c | sort -rn
# churn
git log --since=6.months --numstat --pretty=format: \
  | awk '{a[$3]+=$1} END {for (f in a) print a[f], f}' | sort -rn | head -20
```

The output of §1 is an **order**, not a verdict list. That order is the loop
schedule (§3).

## 2. Why refactor — the reasons that justify the cost

Ranked, because an unranked reason turns into churn.

1. **Change cost sits in the wrong place.** A one-line requirement edits N
   files, or requires reading a 600-line module first. The only reason that
   pays for itself immediately.
2. **Several reasons to change.** Two independent concerns in one module mean
   every future change to either reopens the other's risk surface.
3. **Blast radius.** High fan-in with low cohesion: every caller must be
   re-read to judge whether an edit is safe.
4. **No seam for tests.** If the only way to test a behaviour is to construct
   the world, the boundary is wrong.
5. **Naming lies.** A name that does not predict content makes the codebase
   grep-only.
6. **Duplicated vocabulary.** Hand-copied lists of formats, roles, statuses
   drift from their source of truth; deriving them removes a bug class, not
   just noise.
7. **Onboarding cost.** A newcomer or an agent needs hours to find the entry
   point of a concern.

Non-reasons — say no: line-count vanity ("every file <200 lines"),
a rewrite in disguise ("while we're here, change the algorithm"), style that
the formatter already owns, and any bug.

Rule: if the reason cannot be stated as "an edit of type X costs Y today and
should cost Z", do not start.

## 3. Loop structure — order of chaos

One loop per package or folder. Never one loop for "the codebase".

```text
inventory (§1) → rank folders by chaos → ┌ loop ──────────────────┐
                                         │ review    (§4.2)       │
                                         │ plan      (§4.3)       │
                                         │ perform   (§4.4)       │
                                         │ verify    (§4.5)       │
                                         │ commit (one per step)  │
                                         └───────────┬────────────┘
                                                     │ next folder
                                                     ▼
                              re-inventory → stop when the ranking stops paying
```

**Order of chaos** = sort folders by measured pain (churn × fan-in × size ×
suppression density), take the top one, finish it, re-measure, take the new
top. "Finished" means: reviewed, split, docs updated, gates green, committed.

Two consequences of sorting instead of sequencing: the top fix usually changes
the ranking of everything below it (a shared helper moves into a package, a
caller's shape changes), and a 12-step plan agreed up front is dead by step
three. Re-measure between loops.

Loop budget: one loop should be reviewable in one sitting. If a folder needs
more, split the loop per concern, not per file.

## 4. What each loop contains

### 4.1 Inventory and blast radius

For the target folder: the size list, the suppression list, per-module fan-in,
and separately **who imports it from outside the folder**. That last number
decides whether the move is internal (cheap) or public (needs a compatibility
decision and a stable entry point).

### 4.2 Review — the questions, in order

1. What is this module's one sentence? If it needs "and", it is two modules.
2. What makes it change? List the triggers — a new command, a new format, a
   new provider, a new output mode. Unrelated triggers are boundaries.
3. What is accidentally fenced together? (Console plumbing next to command
   registration; config resolution next to error rendering.)
4. Which helpers are local (one caller, same module) and which are shared
   (three or more callers)? Shared ones are the real seams.
5. Who imports it from a layer that should not know it? A surface layer
   reaching into another surface layer (`mcp/tools` importing a CLI command
   module) is a boundary defect, and the move is when you can finally see it.
   Such a helper belongs in the library both layers already depend on.
6. What does the documentation claim? Every mismatch between a responsibility
   table and the code is a decision someone forgot to write down.
7. What is the public surface? Entry points, `__init__` names, external
   string references in manifests, config, CLI wiring, docs.
8. What shape does the repo already use? Copy it (`commands/` + `args/`
   packages) instead of inventing a parallel structure.

### 4.3 Plan — the target shape

- **One package per concern, named after the concern.** A package earns its
  existence with two or more modules of the same concern, or a documented
  boundary.
- **Keep entry points stable.** Keep the module path that external strings
  point at, so the move does not cascade into manifests and user configs.
- **Split by call graph, not by kind.** "All validators" is not a concern;
  "everything needed to render a page" is.
- **Two grab bags become several real modules.** A `_shared` module is
  typically "which settings a run uses" plus "how a fault reaches the user".
- **Prefer moving to a home over re-exporting.** A shim keeps two homes and
  both rot.
- **A name that crosses a boundary loses its underscore.** Privacy markers are
  for module-local names only.
- **Derive, don't declare**, wherever the vocabulary already exists as an
  enum or schema.

Write the tree and the mapping before moving anything:

| Today | Tomorrow | Reason |
|---|---|---|
| `_console.py` | `runtime/console.py` | process-level console concern |
| `_shared.py` | `settings.py` + `errors.py` | two unrelated reasons to change |
| `_query_helpers.py` | `query/{search,synthesize,render}.py` | three stages of one flow |

### 4.4 Perform — mechanics

Order of operations, each its own commit:

1. **Move**, no edits: `git mv old.py package/new.py` — history follows the
   rename and a reviewer sees 100% similarity.
2. **Split**: move plus cut along the call graph; each new module gets its own
   home.
3. **Rename**: drop the underscore, align the name with the concern.
4. **Repoint**: rewrite the callers, delete the shims.

```sh
git mv src/app/_console.py src/app/runtime/console.py
git diff --cached -M --stat          # confirm rename detection
```

The split technique that survives type checkers:

```python
# 1. copy the SAME import header into every new module
# 2. let the auto-fixer prune what each module does not use
#    (ruff check --fix, goimports, tsc --noUnusedLocals, cargo fix)
# 3. run the type checker: missing names surface immediately
# 4. re-add only the names that were pruned but are genuinely used
```

A hand-written header is a guess; the fixer derives the real one, and the type
checker catches the opposite error (missing) at once.

Pitfalls that cost real time:

- **Ignore rules swallow new directories.** A broad rule for content folders
  (`sources/`, `data/`, `build/`) silently ignores a new package of the same
  name. After creating one, run `git check-ignore -v <new file>` and, if it
  matches, add the narrowest negation (`!src/app/cli/sources/`).
- **Shell string replacement corrupts files.** Chained `-replace`/`sed` runs
  drop control bytes and truncate multi-line imports. Use one surgical edit
  per change, or a script that parses.
- **"Rebuild from version control" undoes the move.** Rebuild an import header
  from the new file, never from the pre-move revision.
- **A big-bang rename hides stragglers.** Grep the old name afterwards across
  source, tests, docs, scripts, ignore files and CI config.
- **A move silently drops annotations.** A `# nocheck`/`# noqa`/`# TODO`
  marker on a moved line disappears when the body is re-typed or rebuilt, and
  the repo's own custom gate is usually the only thing that notices. Grep the
  moved body for markers before pasting, then run the gates.
- **Never mix a move with a behaviour change.** If the split reveals a bug,
  commit the move, then the fix; otherwise the diff cannot be reviewed and the
  bisect cannot find the fix.

### 4.5 Verification — what proves the refactor was safe

Same gates before and after, plus the checks a move specifically needs.

| Check | Question it answers |
|---|---|
| Format / lint / type check | did structure get lost in the move? |
| Repo's own gates (custom checks, doc budgets, parity scripts) | do the repo's invariants still hold? |
| Focused tests for the touched layer | does the behaviour still hold where it matters? |
| Import smoke | does every new module import standalone — no cycle, no missing name? |
| Public-surface parity | are entry points, command names and external strings unchanged? |
| Straggler grep | does anything still point at the old home? |
| Before/after inventory | is the measured problem actually smaller? |

```sh
# import smoke: every changed or new module imports on its own
for f in $(git diff --name-only HEAD~1; git ls-files -o --exclude-standard); do
  case "$f" in *.py) uv run python -c "import ${f%.py}" ;; esac   # adapt per language
done
# stragglers: the old names must be gone
rg "_console|_shared|_query_helpers" || echo "clean"
# parity: diff whatever describes the public surface
app --help > /tmp/help.after.md && diff /tmp/help.before.md /tmp/help.after.md
```

Rules of verification:

- **Do not weaken tests to fit the new shape.** A test that imports an
  internal module gets repointed at the new home; a test that is deleted or
  relaxed so the refactor passes is a regression with a green icon.
- **Capture the surface before starting** (`--help`, a schema dump, an OpenAPI
  export) and diff it afterwards.
- **Verify the smallest scope that can fail** — the layer's own tests plus the
  repo's fast gates. The full suite is a pre-merge gate, not an inner-loop
  tool.
- **State what was not run, and why.**

## 5. Guidelines — the rules that earned their place

1. Split by responsibility, never by size. Size is the symptom; the reason to
   change is the cause.
2. One reason to change per module, and the name states it.
3. Packages before files: the folder is the map, and a flat pile of prefixed
   helpers has no map.
4. Names carry the domain — not an alphabet and not a leading underscore. `_x`
   says "private, don't look", which is exactly wrong for an entry point.
5. Keep public entry points where they are; a refactor must not force every
   consumer to relearn a path.
6. Move, don't re-export. A permanently backward-compatible shim is debt with
   a friendly face.
7. Derive vocabularies from their schema or enum; delete the hand-copied list.
8. One commit per mechanical step (move, split, rename, repoint), each passing
   the gates on its own.
9. Never mix a move with a behaviour change.
10. Measure before and after; if the number did not move, it was a rename.
11. Update the docs in the same commit — responsibility table, index, contract
    references. An undocumented move is a broken map.
12. Stop when the ranking stops paying. Over-splitting buys import noise,
    indirection and file count, not a boundary.
13. Do not refactor what you cannot test or smoke-run; you would have no way to
    know whether it broke.
14. Re-inventory between loops; the tree you finish with is not the one you
    planned against.
15. Treat the move as an audit: duplicates surface (the same constant declared
    twice), reach-throughs surface, lost annotations surface. Delete the
    duplicates instead of relocating them — a move that carries dead weight is
    a missed opportunity.
16. When a parent must learn its children (a sub-app, a registry, a router),
    register them imperatively in the parent with their names and help text.
    Decorators in the children need the parent (cycle) and a side-effect import
    barrel needs an allowlist; one explicit registration keeps the metadata in
    one place.

## 6. Anti-patterns observed

| Anti-pattern | What it looks like | Cost |
|---|---|---|
| Helper grab bag | `_shared.py`, `utils.ts`, `common/` | nobody owns it; it grows by accretion |
| Split by kind | `validators/`, `builders/`, `mappers/` across one flow | every change touches all three |
| Permanent shim | `from new.home import x  # noqa: F401`, kept forever | two homes, both stale |
| Over-splitting | 40-line module behind a 12-line import header | more files to open, no boundary gained |
| Barrel `__init__` | re-exports hiding the real home | imports become unreadable and cyclic |
| Rename without restructuring | `helpers.py` → `support.py` | same missing boundary, new name |
| Move plus fix | the diff that "also" fixes a bug | unreviewable, unbisectable |
| Leaping loop | a 12-step plan agreed up front | step three invalidates the rest |
| Doc drift | responsibility table not updated | the next reviewer trusts a stale map |

## 7. Appendix — evidence from the session that produced this file

Kept for calibration; do not copy the specifics into another project.

- Trigger: a 321-line console entry module, noticed while committing a
  one-line fix. It had five unrelated reasons to change (entry point, BLAS
  thread caps, console hardening, console encoding, fault hooks).
- Split: `launch.py` 321 → 52 lines, plus `runtime/console.py` 199,
  `runtime/faults.py` 164, `runtime/thread_pools.py` 43.
- Regrouping: 14 flat `_`-prefixed modules plus 2 public ones became 5 packages
  (`runtime/`, `tree/`, `sources/`, `workflow/`, `query/`) plus
  `context.py`, `settings.py`, `errors.py`.
  `_query_helpers.py` 474 split into `query/search.py` 215,
  `query/synthesize.py` 185, `query/render.py` 128 along its call graph;
  `_shared.py` 177 split into `settings.py` 138 and `errors.py` 35.
- Blast radius measured before moving: `_shared` 26 files, `_sources` 10,
  `_pipeline_helpers` 8, `_progress` 6, `_post_compile` 5,
  `_query_helpers` 5; the heaviest single consumer was a test file with 13
  references, so the move was mostly internal.
- Folder at review time: 54 modules, 8747 lines. Largest files 647 (28 check
  functions plus rendering), 559, 503 (a 203-line command function behind a
  suppression), 416, 327, 207 — that list is the §3 ranking.
- Verification used: the repo's 12-step gate script (format, lint, type check,
  doc budgets, parity, antipattern scan, smoke), 292 focused tests, then a
  993-test keyword sweep. The full suite was explicitly out of the inner loop.
- Two incidents worth remembering, both in §4.4: an ignore rule swallowing the
  new `sources/` package, and shell string replacement corrupting a file with a
  stray control byte.
- Second loop: `cli/cmds/quality/lint.py` 559 → a `lint/` package of six
  modules (`run` 206, `gaps` 141, `history` 113, `remediate` 108, `formatting`
  63, `reporting` 29) plus a 48-line `__init__` that registers nine
  sub-commands imperatively. The move surfaced three things a size count could
  not: two constants declared twice (deleted, not relocated), an MCP tool
  importing both from the CLI command module (moved to `lint/flow.py` +
  `lint/history.py`), and a dropped `# nocheck` marker that only the custom
  field-consistency gate caught.
