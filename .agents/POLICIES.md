# AGENTS.md — POLICIES

Always applicable. Boundaries, priorities, verification, checklist.

## Priorities

1. Correctness
2. Evidence
3. Safety
4. Minimal changes
5. Consistency
6. Performance

## Boundaries

- NEVER fabricate paths, commits, APIs, config keys, env vars, test results,
  or benchmark numbers. If you don't know, say so.
- NEVER guess at command names, flags, or paths. Read source or run `--help`.
- NEVER add secrets, API keys, or tokens to files. Use env vars.
- NEVER run destructive commands (`rm -rf`, `git reset --hard`,
  `git push --force`) without explicit confirmation.
- NEVER delete or move files without explicit instruction.
- NEVER create temp files in project root. Use dedicated temp dir.

## Change constraints

- Minimal, surgical edits. Preserve existing style.
- No new dependencies without explicit instruction.
- No unrelated refactoring while fixing a bug.
- When changing tests, update only directly affected tests.

## Code quality

Universal defaults. Project-specific standards live in child AGENTS.md.

- **SRP** — one reason to change per module/function. If a name needs "and",
  split it.
- **DRY** — before adding a function/constant/config block, check whether one
  already exists. Extract on the _third_ occurrence (Rule of Three), not earlier.
- **Redundant code** — remove dead branches, unreachable conditions, and unused
  params/imports before finishing.
- **Duplication vs. abstraction** — prefer a little duplication over the wrong
  premature abstraction.
- No new code-quality tooling, patterns, or value-object rules by default — opt
  in per project.

## Completion checklist

- Change solves the stated problem
- Relevant validation ran (or gaps explicitly stated)
- No unintended side effects introduced
- No secrets added or exposed

## DOX authoring (keep AGENTS.md lean)

AGENTS.md and `.agents/` files share limited context. Bloat and duplication
are the failure mode — they waste context on every entry to a subtree and
drift apart as rules change.

### Where a rule lives

Answer one question: **what is the highest (most general) tier that fully
applies?** Put the rule there and nowhere else.

- Applies to **all code in the project**, not one subtree →
  `.agents/POLICIES.md`. This is the default. When unsure, it goes here.
- Meaningful **only inside one subtree** (e.g. framework-specific template
  rules) → that subtree's `AGENTS.md`.
- Just helps navigation → the parent's Child DOX Index, nothing else.

`.agents/POLICIES.md` is the gravity well for universal rules. A subtree
AGENTS.md only contains rules that are meaningless outside that subtree.

### Reference, don't restate

A rule appears in **exactly one file**. Every other file that touches it
uses a **pointer line**: `<topic> — see <file>`. Never copy the rule, the
rationale, or the code example into the second file. If the canonical home
needs a code example, it lives there once — linked to, never duplicated.

### Rule first, rationale second, never third

A DOX line is the rule. If a rationale is costly to rediscover, add one
short clause. Never multi-paragraph explanations or more than one
❌/✅ code pair per rule — those belong in a skill or a `docs/` page, linked
once.

### Content rules

- **No tree views.** Generate on demand with `rg --files | tree-cli --fromfile`.
- **No history.** Git log has it. Record only decisions costly to rediscover
  (in `.agents/HISTORY.md` with commit refs).
- **No TODO lists.** Use issue trackers or beads.

### Size budget

- `.agents/` files: ≤ **120 lines**. They are always-injected context.
- Subtree `AGENTS.md`: ≤ **250 lines**. Loaded cumulatively with every
  parent on entry — the deep-path cost adds up fast.

Exceeding the budget signals the doc is restating instead of pointing, or
hoarding rationale. Cut first; split the subtree only as a last resort.

## Verification

- Run `ruff check` on changed Python files before committing
- Run `uv run pytest tests/unit -x --timeout=30` for unit test coverage
- Run `uv run pytest tests/integration -x --timeout=120` for integration tests
- Run `ty check src/knox` for type checking (mypy is not used)
- Run `uv run python scripts/check_field_consistency.py` to catch hard-coded contract keys
- Run `uv run python scripts/check_antipatterns.py` to catch banned-noqa, magic crops, `try/except ImportError` guards, and `global` keyword usage
- Run `mise run check-all` for the full local enforcement suite

## Response format

Concise and specific. No filler, intros, or restated requirements.
Answer direct questions directly.

For review/debugging/analysis: findings with references, conclusion,
approach. Mention caveats.

## Closeout

1. Run verification
2. Report docs intentionally left unchanged and why

---

## Project-specific Policies — KNOX

### Boundaries

- NEVER hardcode magic numbers, magic strings, or threshold constants in
  production code. Every parameter that affects behavior (token limits,
  failure thresholds, retry counts, timeouts, sizes) must be declared as a
  config field in the appropriate `_KnoxBaseModel` (e.g. `LLMConfig`,
  `BudgetConfig`) with a `Field(validation_alias=...)`. The production
  code reads the value from the config object — never from a bare literal or
  a module-level `Final` constant that duplicates what the config already
  owns. This rule applies even when the default matches the current hardcoded
  value — the field is the single source of truth.
- NEVER modify `.gitignore`, `pyproject.toml`, `ruff.toml`, or CI config
  without explicit user instruction.
- NEVER suppress `PLC0415` (inline imports), `ANN001` (missing return type),
  or `ANN202` (missing return type on public method). These are hard errors —
  no `# noqa` permitted. Refactor the code instead.
- NEVER use `from typing import TYPE_CHECKING` or wrap imports in
  `if TYPE_CHECKING:` blocks. This idiom diverges runtime from type-checker
  behavior, hides module-boundary problems, and adds cognitive overhead.
  Follow the `python-ultimate` skill (`references/type-checking.md`):
  break import cycles structurally (extract shared types to a neutral module,
  use a Protocol, or defer a genuinely runtime-only import to its call site),
  or use `from __future__ import annotations` + forward references.
  `if TYPE_CHECKING:` guards are forbidden in new code and must be removed
  wherever encountered.
- Heavy optional dependencies (docling, torch, transformers, datasketch,
  scipy, crawl4ai, pydantic-ai, MCP SDK, zvec) MUST be lazy-loaded so that
  CLI startup (`--help`, `--version`) stays fast. Use `lazy-loader`
  (SPEC-0001) in `__init__.py` via `lazy.attach(__name__, submod_attrs={...})`
  to expose heavy submodules as on-demand attributes. For heavy classes used
  only inside a function body, use the parser registry's `get_backend()` or
  equivalent factory pattern — never import the class at module scope.
  See `src/knox/preprocessing/parsers/__init__.py` for the canonical pattern.
- NEVER persist absolute filesystem paths inside `.knox/` artifacts
  (JSON, JSONL, SQLite DBs, TOML). Paths in every persisted record —
  provenance, references (`citing_source`), assets (`source`), dedup,
  memory, search index, organize reports, compile state, wiki cache — must
  be **relative to the vault `base_dir`** and POSIX-normalized
  (forward slashes). The vault must be relocatable: copying the folder to
  another machine or path must not break a single stored reference.
  In-memory `Path` objects may be absolute (convenient for I/O); the
  relativization happens at the **persistence boundary** via
  `WikiPipeline._rel_path()` (or its standalone equivalent). Absolute
  paths leaking into a persisted field is an antipattern and a bug.
- Use `./tmp` (gitignored) or `tempfile.TemporaryDirectory()` for temp files.

### Error handling — prefer "let it burn"

The default is to **propagate exceptions**. Surface real failures instead of
hiding them. The codebase historically over-used `except: pass` and
`except: return None`, converting real errors (bad API keys, corrupt state,
missing deps, disk failures) into silent `None` values — forcing every caller
into `if x is None` chains and producing misleading messages.

`None` may only mean a thing was **absent** (lookup miss, optional value
unset), never that an operation **failed**. Returning `None` for a corrupt
file is the antipattern.

The only valid `try/except` sites:

1. **Trust boundary** — translate a low-level error to a user-facing message
   at the CLI/MCP top level. Catch narrowly, report the real cause.
2. **Optional-feature probe** — narrow to the _specific_ exception
   (`ImportError`, `FileNotFoundError`), never bare `Exception`.
3. **Best-effort enrichment** — per-item catch that _records_ the failure
   (SKIP result, warning log) and continues a batch.
4. **Cache miss** — corrupt/unreadable cache entry treated as a miss is
   correct, BUT narrow to `(json.JSONDecodeError, OSError)` and log at
   `WARNING` so corruption stays visible.

Forbidden:

- `except Exception: pass` / `except Exception: return None`
- `# noqa: BLE001` / `# noqa: S110` to silence a bare-except lint
- Muting an exception then returning `None` to mean failure
- Catching a corrupt internal-state file (wiki cache, tier state, compile
  state) and silently resetting — corruption is a bug; **raise**.

Rule of thumb: if removing the `try/except` surfaces a real error to someone
who needs to see it, remove it. If the catch genuinely serves one of the four
valid sites, narrow the exception and make the failure visible.

### Implementation Plans

- Implementation plans, refactoring plans, and design docs go in
  `.agents/plans/<feature>/` — NOT under version control (gitignored).
- Name plans descriptively and prefix them with phase numbers, so that the file order
  corresponds to implementation order, e.g. `01-citation-extraction.md`, `02-refactoring-llm-agent.md`.
- NEVER commit implementation plans to git. They are highly-mutable working artifacts,
  not shipped documentation.
- Shipped docs (user-facing guides, API references) go in `docs/` and
  ARE committed.

### User Preferences

- Concise, direct answers with no filler or intros
- Use `uv run` for all Python execution, never direct `python` calls
- English-only for code, docs, and commits
- Conventional commits format (`feat:`, `fix:`, `chore:`, `docs:`, etc.)
