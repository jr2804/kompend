# Refactoring a growing codebase — inputs for a skill

Status: raw material for a future `/refactor` skill, collected 2026-11-06 from
one long session in this repository. Everything above the appendix is
project-agnostic on purpose; the appendix holds the evidence that produced it
and is there for calibration, not for copying.

## 0. What the skill must encode

| Question       | Answer                                                                                                            |
| -------------- | ----------------------------------------------------------------------------------------------------------------- |
| Trigger        | an explicit request ("refactor/split/clean up X") or a measured signal from §1 that crosses the repo's own median |
| Input          | one folder, one concern — never "the codebase"                                                                    |
| Output         | commits (one per mechanical step), a before/after inventory, updated docs                                         |
| Stop condition | the §1 ranking stops paying (see §5.12)                                                                           |
| Forbidden      | mixing a move with a behaviour change; weakening tests to pass; refactoring what cannot be run                    |

Not a trigger: "the code looks ugly", a pending feature (build it, then
refactor), a bug (fix it first, separately).

## 1. Status — what "needs refactoring" actually looks like

Measure, never opine. Compare against the repo's own median, not an absolute
truth: a 300-line module is unremarkable in a repo whose median is 280 and a
smell in one whose median is 40.

| Signal               | Measure                                                      | Red flag                                          | Why it matters                                       |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------- | ---------------------------------------------------- |
| File size            | lines per leaf module                                        | far above the repo median, or a leaf >300         | the file is a container, not a concept               |
| Function size        | statements per function                                      | >60, or a suppression (`# noqa: PLR0915`, `C901`) | nobody can hold it in working memory                 |
| Suppression density  | suppressions per file                                        | more than a couple                                | each one is a rule the code cannot meet              |
| Naming               | `utils`, `helpers`, `_shared`, `common`, `misc`, `_anything` | any                                               | the name says "not my job", so nobody finds anything |
| Reasons to change    | list the module's unrelated "and also" claims                | >1                                                | the definition of a missing boundary                 |
| Fan-in               | importing files per module                                   | top decile of the repo                            | blast radius of every edit                           |
| Churn                | commits touching the file over 6 months                      | top decile                                        | where the pain actually is                           |
| Test mirror          | test file size vs module size                                | test ≥ module, or a 1:1 giant                     | the module has no seam                               |
| Duplicate vocabulary | hand-written lists that mirror an enum/schema                | any                                               | drift by construction                                |
| Doc drift            | responsibility tables vs the code                            | mismatch                                          | the map lies, so reviews miss things                 |

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
- **One module per registered command, when the registry imports lazily.** The
  command tree already loads a module on first use, so one file per command
  narrows what each command pulls in: `kompend index` no longer imports the
  compile path, the distill workflow or the post-compile actions. A module
  holding four commands cannot offer that.
- **A wrapper that only forwards keywords is not a split.** Moving a 20-argument
  call into a 20-argument helper neither reduces statements nor creates a seam;
  it duplicates the signature. Extract a phase only when it names a step.
- **A command module is a surface; an engine belongs under the package tree.**
  When a module's body answers questions about the world (is this host ready, is
  that model reachable, what does this file contain), it is a library that
  happens to have a CLI: move it to `kompend/<domain>/` and leave the command
  with rendering and argument plumbing. Splitting it into
  `cli/cmds/<domain>/<thing>/` keeps it unreachable for every other surface,
  and the second surface then imports a command module — the boundary defect
  this loop exists to remove.
- **A name that crosses a boundary loses its underscore.** Privacy markers are
  for module-local names only.
- **Derive, don't declare**, wherever the vocabulary already exists as an
  enum or schema.

Write the tree and the mapping before moving anything:

| Today               | Tomorrow                              | Reason                          |
| ------------------- | ------------------------------------- | ------------------------------- |
| `_console.py`       | `runtime/console.py`                  | process-level console concern   |
| `_shared.py`        | `settings.py` + `errors.py`           | two unrelated reasons to change |
| `_query_helpers.py` | `query/{search,synthesize,render}.py` | three stages of one flow        |

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
- **Check the working tree is yours before a repo-wide formatter.** Another
  session, a background job or a half-finished stash can own files in the same
  checkout, and `ruff format .` / `check --fix` / `gofmt ./...` will rewrite
  them mid-edit. `git status` first, then scope the command to the paths the
  loop owns; gates that scan the whole tree stay red until their work lands,
  which is not your regression.
- **A hand-written record re-lists its own fields.** `__init__` plus `to_dict`
  over four attributes is a `@dataclass(frozen=True, slots=True)` plus one
  method. The two sides drift the first time a field is added, and only a
  field-consistency gate notices. When a move touches such a class, declare it
  as a dataclass instead of relocating the constructor.
- **A `dict[str, Any]` boundary parameter is a record nobody named.** A config,
  an options bag or a metadata mapping walked with `cfg.get("dotted.key", 300)`
  carries a literal default that no schema knows (the real default was 120) and
  a key string that duplicates a constant — sometimes one that does not exist
  yet. Read the typed model at the boundary (`resolved.kompend.<section>`) and
  keep the mapping only where the data is genuinely dynamic.
- **Never mix a move with a behaviour change.** If the split reveals a bug,
  commit the move, then the fix; otherwise the diff cannot be reviewed and the
  bisect cannot find the fix.

### 4.5 Verification — what proves the refactor was safe

Same gates before and after, plus the checks a move specifically needs.

| Check                                                         | Question it answers                                                  |
| ------------------------------------------------------------- | -------------------------------------------------------------------- |
| Format / lint / type check                                    | did structure get lost in the move?                                  |
| Repo's own gates (custom checks, doc budgets, parity scripts) | do the repo's invariants still hold?                                 |
| Focused tests for the touched layer                           | does the behaviour still hold where it matters?                      |
| Import smoke                                                  | does every new module import standalone — no cycle, no missing name? |
| Public-surface parity                                         | are entry points, command names and external strings unchanged?      |
| Straggler grep                                                | does anything still point at the old home?                           |
| Before/after inventory                                        | is the measured problem actually smaller?                            |

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

| Anti-pattern                 | What it looks like                                     | Cost                                   |
| ---------------------------- | ------------------------------------------------------ | -------------------------------------- |
| Helper grab bag              | `_shared.py`, `utils.ts`, `common/`                    | nobody owns it; it grows by accretion  |
| Split by kind                | `validators/`, `builders/`, `mappers/` across one flow | every change touches all three         |
| Permanent shim               | `from new.home import x  # noqa: F401`, kept forever   | two homes, both stale                  |
| Over-splitting               | 40-line module behind a 12-line import header          | more files to open, no boundary gained |
| Barrel `__init__`            | re-exports hiding the real home                        | imports become unreadable and cyclic   |
| Rename without restructuring | `helpers.py` → `support.py`                            | same missing boundary, new name        |
| Move plus fix                | the diff that "also" fixes a bug                       | unreviewable, unbisectable             |
| Leaping loop                 | a 12-step plan agreed up front                         | step three invalidates the rest        |
| Doc drift                    | responsibility table not updated                       | the next reviewer trusts a stale map   |

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
- Third loop: `cli/cmds/pipeline/pipeline.py` 555 → `parse.py` 51,
  `index.py` 84, `ingest.py` 140, `compile.py` 380 (of which 35 lines are the
  command's keyword options and 5 helpers carry its phases). The old
  `compile_cmd` carried `# noqa: PLR0915`, which is now gone: extracting
  `_publish_pending_notes`, `_start_progress`, `_run_pipeline`, `_post_compile`
  and `_report_failure` removed the suppression rather than widening it. The
  move also exposed two pieces of dead code — `time.monotonic() - ingest_start`
  computed and discarded, and an `if isinstance(exc, LLMOutputError)` whose two
  branches built the same string — and one duplication: `--attr key=value` was
  parsed in two modules with two different messages, now one `parse_attrs` in
  `cli/settings.py`. A cross-command inconsistency was found and deliberately
  NOT fixed here (`parse` never passed `config_file` to `resolve_config`,
  so `--config-file` was ignored): a behaviour fix is its own commit.
- Fourth loop: `cli/cmds/quality/doctor.py` 746 → a library of seven modules
  (`run_doctor` in `diagnostics/runner.py` 134, `providers.py` 312,
  `models.py` 279, `environment.py` 82, `base.py` 33, `capabilities.py` moved
  unchanged) plus a `doctor.py` that is 51 lines of rendering. The size was
  never the argument — the argument is that the command module answered
  questions about the host, the provider endpoints and five model properties,
  and that the MCP `doctor` tool had to import a CLI command module to get
  them. The loop produced the antipattern inventory the owner asked for: a
  hand-written record class, `cfg: dict[str, Any]` with a stale literal default
  and a hand-written dotted key, `LLMConfig(**cfg.get(...).get(...))`
  dict-splatting, a `# nocheck: enum-literal` hiding a provider id, three
  copies of provider-string parsing across the repo, a dead offload branch, and
  a check documented as the only catalogue refresher that `run_doctor` never
  called (left unwired: that is a behaviour change, reported instead).
- Fifth loop (bead knox-xapn): the fourth loop's finding "three copies of
  provider-string parsing across the repo" landed as `kompend.llm.model_id`
  (43 lines: `provider_id_of`, `model_slug`, one private `_split_model_id`;
  bare model → `None` provider), and four call sites now consume it —
  `llm/user_providers.py`, `llm/debug.py`, `diagnostics/providers.py`,
  `rag/vector_store.py`. The move surfaced what the size count could not: the
  copies disagreed about _which_ separator wins, not only about the edge
  cases. The doctor pair checked `/` before `:` by code order, not in-string
  order, so it mis-split a colon-first string holding a slash model
  (`ollama:hf.co/...` → provider `ollama:hf.co`); `rag` checked `:` first and
  was right; the two `llm/` sites split on `/` only, so `effective_workers`
  looked up `[providers.<name>]` with the raw prefix and silently ignored
  `max_connections` for `ollama:model:tag`, while the debug slug kept the
  provider for one spelling and stripped it for the other. Each disagreement
  was pinned red in `tests/unit/test_model_id.py` before the repoint, then
  resolved to the union rule — the first separator in string order, which is
  `models._split`'s rule minus its raise on the bare form. `models._split` and
  `context_window._split_model` stay: they are build-path splitters with
  stricter no-separator contracts (raise / empty provider), and `rag`'s
  known-provider gate is its own contract, not a parse. The move and the
  behaviour fixes are separate commits (`d62e23cd`, `5d5d7eb6`), and the
  doctor's `ValueError` for an unknown provider prefix survives unchanged,
  because the call site keeps its own known-provider check and only the split
  moved. The lesson is the one the fourth loop already paid for: the size was
  never the argument, the four answers to "which provider is this model" were.
- Sixth loop (bead knox-2xbd): `core/pipeline/stages/` held six underscore-prefixed
  modules, and `stages/_parse.py` was 769 gross / 504 net lines answering three
  questions — how Tier 1 dispatches, what happens to one source, and how its parse
  and chunk caches are read and written. The underscores went first, as one pure
  `git mv` (`_parse` → `parse`, plus `_pipeline`, `_factory`, `_index`, `_resolve`,
  `_compile`), with every import site repointed in the same commit, because the
  modules were internal to `core/pipeline` and no public path depended on them.
  Then `_parse.py` split along its call graph into `parse.py` (365 / 216: dispatch,
  worker pool, dedup gate, attachments, `build_resolver_chain`), `parse_sources.py`
  (267 / 188: `_process_single_source`, `_parse_source`, privacy filter, dedup
  warning) and `parse_store.py` (194 / 117: the six cache functions), which took the
  subtree from 1186 net lines over 7 files with one file past the 350 threshold to
  1203 net lines over 9 files with none. The move surfaced the duplicate the bead
  only half-named: the stage carried a local `cache_key(source_file)` re-deriving
  `source_key(stem, content_hash_prefix(bytes))` while `parse_cache.py` next door
  already exported `cache_key` _and_ `parse_cache_is_current`. The extraction was
  therefore a deletion — call the canonical key, drop the local copy — committed on
  its own before the split, with the key proven identical (`note-ba7816bf` from both
  forms). No shim re-exports were left: `ruff` pruned the ones staged as a
  transition and the tests were repointed at the new homes, including the
  `caplog` logger string, which had to split into `_PARSE_SOURCES` and
  `_PARSE_STORE` because the warnings moved with their call sites. Every function
  body was compared against the pre-split file (16 defs, zero differences), so the
  `HeavyParseGuard` budget, the `Skipped <file>: <reason>` contract, the reason
  codes, the ledger rows and the `cancel_futures` shutdown are untouched. The lesson
  is that a 504-line stage file was not the argument; the argument was that three
  questions shared one module and one of them had already been answered next door.
- Seventh loop (bead knox-n869): `output/vault.py` 879 gross / 541 net and
  `output/markdown.py` 563 / 325 became `page.py` 432 / 216 (the OKF frontmatter
  plus every body transform), `page_paths.py` 61 / 15 (page layout vocabulary:
  `bundle_path`, `page_stem`, `_folder_for_kind`), `surfaces.py` 379 / 260
  (per-source summaries, `references.md`, `quality.md`), `vault.py` 373 / 207 (the
  vault-wide loop and the format hooks) and `markdown.py` 335 / 204 (the markdown
  format). The size was never the argument — the argument is that one file answered
  five questions at once (what frontmatter is, what a page body looks like, where a
  page lands, what the non-page surfaces are, and how a vault is swept), so a change
  to the description cap and a change to stale-file clearing fought over the same
  file. The assembly is mixins, not inheritance chains: `VaultWriter` is
  `OkfPageMixin + PageLayoutMixin + GeneratedSurfaceMixin + BaseOutputWriter`,
  which keeps the external surface (`VaultWriter.okf_frontmatter`,
  `_folder_for_kind`, `_KIND_FOLDERS`, `okf_description`) resolving through
  inheritance while `output/pages/index.py` imports the two mixins it actually
  uses instead of the vault writer it never instantiates.
  The move surfaced four things the line count could not: (1) `vault.py`
  carried a byte-identical copy of `markdown.py`'s `WIKILINK_RE` — deleted, not
  relocated; (2) `vault.okf_frontmatter_yaml` was a forwarding wrapper whose
  docstring claimed capture writers needed it to dodge this module's
  dependencies, but `capture/capture.py`, `distill.py` and `learn.py` all
  import `kompend.output.frontmatter` directly — deleted, and that removal is
  a deliberate public-surface change recorded in its own commit; (3)
  `_write_source_summaries` and `_write_references_page` each enumerated the
  corpus sources with the same `sorted(... if p.is_file() and not
is_protected_source(...))` expression — now one `source_keys()`, and the
  dead `_source_name(key)` statement whose return value was discarded went with
  it; (4) three page-path templates disagree for an empty slug (`vault`'s has
  the `untitled` fallback, `MarkdownWriter._write_index`'s does not,
  `pages/index.py`'s adds a prefix). They were deliberately left divergent —
  unifying them changes the write path for empty slugs, which is a behaviour
  change and its own bead — and the divergence is documented in the
  `page_paths.py` module docstring. The other finding is reported, not fixed:
  `VaultWriter.__init__` falls back to bare `OutputConfig()` / `DisplayConfig()`
  (schema defaults, not resolved config); its docstring claims
  `output.max_related_links` where the real dotted path is
  `compile.output.max_related_links`, and the capture and MCP call sites pass
  only `description_max_chars`, so a configured `max_related_links` is silently
  ignored.
- Eighth loop (bead knox-6aoi): `rag/sqlite_vec_store.py` 1025 gross / 577 net
  became a package of six implementation modules plus a public surface —
  `sqlite_vec/store.py` 532/327, `query.py` 237/104, `manifest.py` 159/81,
  `index.py` 135/96, `schema.py` 122/51, `filters.py` 72/34, `__init__.py`
  14/1 — with the old module path kept as a 14-line compatibility re-export.
  Subtree went 1846/914 to 2106/915 net: the split costs one line of glue
  and buys a 43 % cut in the largest file, and nothing in the subtree is
  over the 350-net threshold any more.
  What the size count could not see: the bead's step 3 ("decide ONE home for
  `matches_attr_filter` … delete the other copy") was already settled — the
  one Python predicate is `search/filters.py:25` and the store only ever held
  its SQL translation, with `TestFilterParityWithCanonicalPredicates` pinning
  them together, so there was nothing to delete and the step was a no-op. The
  seam list was also stale by one: the file held an embed-and-insert job the
  target shape did not name, which is why `index.py` exists. And the old
  module path turned out to be a _published_ one — `docs/integrations/
open-webui.md` shows `from kompend.rag.sqlite_vec_store import
SqliteVecWikiStore` — so the shim stayed rather than being deleted, and
  four in-repo callers in other lanes' files still import the old path (they
  work, but they now break the "import from the package root" rule; reported,
  not fixed). Two smaller things a mover should expect: the copied
  `# ruff: noqa: S608` marker was needed in all five SQL modules and then
  became dead in `store.py` once its last interpolated statement left — ruff's
  RUF100 caught it, which is why the last commit is a two-line style fix; and
  the DOX line "no module below `store.py` opens a file" was wrong on arrival
  because `manifest.py` must open `kompend.db` before any store exists, so
  the DOX was corrected instead of leaving a rule the code contradicts.
- Ninth loop (bead knox-jjt5): `lint/structural.py` 1363 lines / 4 noqa became
  `lint/structural/` — `attachments.py` 403, `links.py` 244, `references.py`
  175, `structure.py` 171, `markdown.py` 166, `sources.py` 142, `__init__.py`
  84 (registry and public surface only), `images.py` 76, `metadata.py` 64 — 9
  files / 1525 gross. The largest file is 70 % smaller; the 162 extra lines
  are one import header per module plus the registry that replaced the old
  hand-written if-chain. The 14 rule names, messages and their execution
  order are unchanged — the bead declared them contracts, so each family
  module exports an ordered tuple (`IMAGE_RULES`, `LINK_RULES`, …) and
  `__init__.py` owns the one ordered
  `STRUCTURAL_RULES` list (image, link, metadata, page structure, source,
  attachment) plus `RULE_BY_NAME`.
  What the size count could not see: (1) the split is by rule _family_, not
  by rule — `links.py` and `images.py` both index the same nodes and
  `references.py` and `markdown.py` both read page text, so a file per rule
  would have multiplied the shared walkers instead of removing them; (2) the
  four suppressions are properties of the algorithm, not of the container —
  S603 marks the one internally constructed subprocess call
  (`markdown.py:147`) and PLR6301 marks three `fix` methods that genuinely
  ignore `self` (`images.py:61`, `links.py:99`, `metadata.py:43`), so the
  split could not and should not drop them; (3) the bead's target shape said
  `lint/rules/structural/`, but `lint/structural/` shipped because the parent
  already holds non-rule modules (`base.py`, `runner.py`, `flow.py`), so a
  `rules/` level would only add a synonym for "the structural rules".
