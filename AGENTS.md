# DOX framework

- DOX is a highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits
- This document is the **root DOX rail** for the KNOX repository

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs
  must stay understandable from the nearest applicable AGENTS.md plus every parent
  AGENTS.md above it

## Read Before Editing

1. Read this root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path,
   read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for
   repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child
   doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session
before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child
index changes. Update child docs when parent changes alter local rules.
Remove stale or contradictory text immediately. Small edits that do not change
behavior or contracts may leave docs unchanged, but the DOX pass still must
happen.

## Style

All AGENTS.md files are loaded into the agent's context window on every
applicable task. Every redundant word, duplicated rule, or overly verbose
description wastes context and dilutes signal.

- **No redundancy.** A rule must live in exactly one AGENTS.md. If a concept
  applies to multiple scopes, put it in the nearest common parent doc, not in
  every child. Cross-reference with a one-liner rather than copying text.
- **Precise and brief.** Say what the agent must know in as few words as
  possible. Favor bullet lists over prose paragraphs. Favor tables over
  bullet lists when comparing multiple items.
- **Delete over explain.** Remove stale or contradictory text immediately.
  Do not preserve old rules with a deprecation note — delete them.
- **Every line earns its place.** If a sentence could be removed without
  changing the agent's behavior, remove it.
- **Child docs are local only.** A child doc must not repeat rules from its
  parent. If it needs to reference a parent rule, write "See root AGENTS.md:
  [rule name]" — not the full text.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences,
  durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child
  DOX Index
- Each parent explains what its direct children cover and what stays owned by
  the parent
- The closer a doc is to the work, the more specific and practical it must be

### When to create a child AGENTS.md

Create a child AGENTS.md only when **all** of these hold:

1. **Durable boundary** — The directory has a stable, recurring purpose, not a
   one-off task or transient structure.
2. **Distinct rules** — The work done here follows rules materially different
   from the parent's rules.
3. **Sufficient density** — Rules are numerous or nuanced enough to justify a
   separate file rather than a bullet in the parent.
4. **Code directories only** — Children are for code directories where agents
   do substantial work and frequently repeat mistakes. Test code counts as
   code — tests have distinct patterns and antipatterns that warrant a local
   doc, especially to prevent file-creation errors.

**Counterexamples** that do not warrant child docs:

- `docs/` — documentation format rules fit in a paragraph in the parent
- `.forgejo/` — CI workflow rules are self-documenting in YAML
- `scripts/` — utility scripts with no distinct agent-facing rules
- `.config/`, `.vscode/` — tool configuration, not agent work boundaries

When in doubt, **keep rules in the parent**. A child doc is a commitment to
maintain it. Delete or merge child docs when their rules become sparse or are
absorbed by the parent.

### KNOX hierarchy

```text
AGENTS.md (root)               ↤ project-wide rules
├── src/knox/AGENTS.md         ↤ Python source: text-in-code, antipatterns
│   ├── cli/AGENTS.md          ↤ CLI layer: thin shell over core
│   ├── core/AGENTS.md         ↤ Core foundation: models, paths, config, pipeline
│   │   ├── core/config/AGENTS.md     ↤ Config: resolution, precedence, TOML
│   │   ├── core/models/AGENTS.md     ↤ Data models: enums, chunks, trees, wiki
│   │   ├── core/pipeline/AGENTS.md   ↤ Pipeline: orchestrator, config, state, ingest
│   │   ├── core/resolution/AGENTS.md ↤ Tier 3: entity registry, xref checker
│   │   ├── core/cache/AGENTS.md      ↤ Content-addressed cache store
│   │   ├── core/templates/AGENTS.md  ↤ Layered Jinja2 template loading
│   │   ├── core/trust/AGENTS.md      ↤ Actor identity
│   │   ├── core/reporting/AGENTS.md  ↤ CLI observability reports
│   │   └── core/utils/AGENTS.md      ↤ Pure utilities: time, exceptions
│   ├── memory/AGENTS.md       ↤ Memory: query store + compile lifecycle
│   └── llm/AGENTS.md          ↤ LLM client: errors, failover, custom providers
│       └── providers/AGENTS.md ↤ Provider registration rules and patterns
└── tests/AGENTS.md            ↤ testing conventions, test antipatterns
```

---

## KNOX Project Rules

### Project

**KNOX** = Knowledge Organizer for eXtended documents.

A parser-agnostic, structure-preserving, LLM-aware knowledge base compiler.
Follows a 5-tier pipeline:
Preprocessing → Chunking → Resolution → Compilation → Output.

### Priorities

If rules conflict, lower-numbered priority wins:

1. Correctness
2. Evidence
3. Safety
4. Minimal changes
5. Consistency
6. Performance

### Boundaries

- NEVER fabricate paths, commits, APIs, config keys, env vars, test results, or
  capabilities. State gaps explicitly.
- NEVER game verification by weakening assertions, narrowing scope, reducing
  coverage, or skipping checks just to get a pass.
- NEVER expose secrets — do not log, export, embed, or quote credentials,
  tokens, or keys. If encountered mid-task: (1) do not include any part of the
  secret in your response; (2) immediately stop all work; (3) report the file
  path and line range (not the secret value) to the user; (4) discard any
  uncommitted changes that would expose the secret. Do not commit or push
  before reporting.
- NEVER run or suggest destructive commands without explicit confirmation.
- Be direct. Avoid flattery, filler, and agreeing with incorrect premises.

### Uncertainty

- Ask before acting when intent is materially ambiguous.
- Ask before choices that change behavior, API/UX, naming, auth, dependencies,
  or compatibility.
- Prefer one targeted question. When bundling, ensure each question can be
  answered independently.
- Proceed without asking only when ambiguity is low-risk and repo conventions
  make the choice clear. State the assumption briefly.
- **Exception — Session Completion overrides the ask-before-persistence rule
  for end-of-session pushes.** The mandatory push workflow (see Completion) is
  autonomous and does not require confirmation. All other Uncertainty rules
  still apply.

### Evidence

Gather evidence proportional to risk.

- Trivial low-risk edit: inspect the target file and adjacent context.
- Medium-risk change (logic, algorithm, multi-file refactor with no API or
  dependency impact): trace the direct execution path and check existing tests
  for the affected area before editing.
- Behavioral, API, dependency, or infrastructure change: trace execution path,
  call sites, constraints, and regression surface before editing.
- Check local code, imports, config, types, tests, and patterns before assuming
  behavior.
- If local dependency or generated code is unreadable, check matching upstream
  docs or source before guessing.
- Prefer external verification over self-review. A fresh test beats re-reading
  your own code.
- State uncertainty when something cannot be confirmed.

Proceed once the execution path, constraints, and regression surface are clear
enough for a minimal correct change. If not, ask or report the gap.

### Workflow

1. Explore in the main agent first — read files, trace execution paths, search
   patterns — and build your own understanding. Do not delegate before you have
   seen the data.
2. Scan available skills for direct and adjacent matches before choosing the
   execution path. When in doubt, load the skill and check.
3. Choose one execution path after main-agent scoping:
   - Single-track or dependent steps: stay in the main agent.
   - Small reads or searches: use parallel tool calls in the main agent.
   - 2+ independent tracks: launch all subagents in the same response.
   - Use 2+ subagents or none. NEVER launch exactly 1 subagent.
4. Synthesize findings and re-read target files if context is stale.
5. Implement the smallest correct change.
6. Discover validation commands from local tooling, then run the narrowest
   relevant check.

Workflow compression means skipping steps 1–2 (explore and scan skills) and
proceeding directly to step 5 (implement). This applies only to coupled,
single-track work where the next step depends on the current finding.

For review, debugging, or analysis requests, do not make code changes unless
the user explicitly asks for a fix after seeing your findings. Report findings
first; wait for instruction before editing.

### Subagents

Use 2+ subagents or none. NEVER launch exactly 1 subagent.

The main agent is a builder, not a dispatcher. Work first, delegate second.
Use subagents proactively, but only after scoping has split the work into
tracks ready for parallel execution.

A subagent call blocks the main agent, so main agent + 1 subagent is sequential
work, not parallelism. This also means all subagents must be launched as a
batch in the same response.

Before launching subagents, confirm: (a) main-agent scoping is complete,
(b) work splits into 2+ fully independent tracks, (c) each track has a concrete
return format. If all three are true, launch all tracks in one response.
Otherwise, stay in the main agent.

- Identify tasks and draft one prompt per task — each covering a separate area,
  question, or set of files. Keep scoping in the main agent until you have 2+
  prompts ready.
- Each track must complete without the results of the others. If a track
  depends on another's findings, handle it in the main agent.
- Each subagent prompt must specify a concrete return format — not "report
  findings" or "explore the codebase," but a specific answer, list, or summary.
- Subagents are bound by the same Boundaries, Evidence, and Safety rules as the
  main agent. The concrete return format is in addition to, not instead of,
  these constraints.
- Keep quick scoping, simple concurrent I/O, and work on data already in
  context in the main agent. Use parallel tool calls when helpful.
- Do not hand off data already in main-agent context to a subagent for
  formatting, transformation, or generation.
- After the batch returns, synthesize results and use the main agent only for
  narrow gap-filling before implementation.

### Change Constraints

- Do exactly what was asked. Do not expand scope without clear reason.
- Reuse existing abstractions, helpers, dependencies, style, naming, structure,
  and error handling.
- Prefer the smallest viable change. Do not modify working code without clear
  justification.
- Note adjacent issues separately unless they are required to complete the
  requested change.
- Add dependencies only when necessary. Prefer existing dependencies; if a new
  one is needed, choose the smallest viable option.

### Safety & Infrastructure

- Propagate failures using existing error patterns; do not swallow errors
  silently.
- Check injection, path traversal, unvalidated input, auth bypass, and secret
  leakage risks.
- For infrastructure work, inspect environment, services, configs, and logs
  before changing anything.
- Validate config before reload or restart; prefer reload when safe.

### Git & PRs

- Commit only when explicitly requested, except during Session Completion
  (see Completion section).
- Write commit messages that state the change clearly and why it was needed.
- Keep PRs small and scoped to one concern.
- Do not force-push to main/master.
- Do not use `--no-verify` or `--no-gpg-sign`.
- Every push to `main` triggers an auto-release (`calver` tag + Forgejo
  release). Add `[skip release]` to the commit message to push without
  releasing — use for doc-only, chore, or beads-sync commits.

### Completion

Before declaring completion, confirm:

- The change solves the stated problem
- Relevant validation ran or gaps are stated
- No known unintended side effects were introduced
- No secrets were added or exposed

### Response Format

Be concise and specific by default. No filler, intros, or restated requirements.
Answer direct questions directly when possible.

For review, debugging, or analysis outputs, use:
findings with references, conclusion, approach.
Mention caveats and unverified risks.

### Conventions

- English-only for all code, docs, and commits
- Use `uv run` / `uv run -m` for all Python execution; never call `python`,
  `python3`, `python -m`, `python3 -m`, or `pip` directly
- Conventional commits (`feat:`, `fix:`, `chore:`, `docs:`)
- All parsers implement `BaseParser`; all chunkers implement `BaseChunker`
- Online sources (URLs) go through `downloaders/` before `parsers/`
- MCP server is a thin wrapper — no business logic duplication
- Documentation uses standard Markdown. Internal links use relative paths
  (`../architecture/index.md`). Diagrams use Mermaid in fenced code blocks.
- Release versioning follows `YYYY.M.D` / `YYYY.M.D.N` calver format
  (no zero-padding, no `v` prefix). See `.forgejo/workflows/release.yml`.

### Path Constants — Single Source of Truth

All on-disk artifact names (directory and file names under the project root)
are defined as `Final[str]` constants in **`src/knox/core/paths.py`**. No other
module may hard-code these strings.

**Rule:** The string `".knox"`, `"cache"`, `"config.toml"`, `"vectors"`,
`"memory.db"`, `"last_wiki.json"`, `"search.db"`, etc. must **never** appear as
a literal in any `.py` file outside `paths.py`. Always import from
`knox.core.paths`.

| Constant | Value | Usage |
|----------|-------|-------|
| `KNOX_DIR` | `".knox"` | Root metadata directory |
| `CACHE_SUBDIR` | `"cache"` | Cache component within `KNOX_DIR` |
| `CACHE_DIR` | `".knox/cache"` | Full relative cache path (for DEFAULTS) |
| `CONFIG_FILE` | `"config.toml"` | Scaffolding config filename |
| `KNOX_CONFIG_FILE` | `".knox/config.toml"` | Project-level config relative path |
| `WIKI_CACHE_FILE` | `"last_wiki.json"` | Compiled wiki cache |
| `REVIEW_DIR` | `"review"` | Review queue directory |
| `VECTORS_DIR` | `"vectors"` | Embedding vector store |
| `SEARCH_DB_FILE` | `"search.db"` | SQLite search index |
| `DOWNLOADED_FILE` | `"downloaded.json"` | URL download manifest |
| `MEMORY_DB_FILE` | `"memory.db"` | SQLite memory store |
| `MEMORY_LOG_FILE` | `"memory.jsonl"` | Memory append-only log |
| `MEMORY_STORE_FILE` | `"memory.json"` | Memory JSON export |
| `PROVENANCE_DIR` | `"provenance"` | Provenance trace directory |
| `VERIFICATION_FILE` | `"verification.json"` | Verification results |
| `PRIVATE_FILE` | `"private.json"` | Private page manifest |
| `COMPILE_STATE_FILE` | `"compile_state.json"` | Tier state checkpoint |
| `LINT_HISTORY_FILE` | `"lint_history.jsonl"` | Append-only lint history log |
| `STATE_FILE` | `"state.json"` | General state file |

**Configurable directory names** (`"sources"`, `"wiki"`) live in `DEFAULTS`
under `["paths"]["sources_dir"]` and `["paths"]["output_dir"]` — not in
`paths.py`, because users can override them.

**Project root naming:** The project root directory is always called
`base_dir`. Never use `target_dir`, `resolved_base`, `project_root`, or
`sources_dir.parent` to refer to the project root.

#### VaultPaths — resolved path resolution

`VaultPaths` (in `paths.py`) is the canonical way to resolve concrete paths
inside a vault. It takes a `base_dir` and optional dir-name overrides,
exposing every artifact path as a `@property`:

```python
from knox.core.paths import VaultPaths

paths = VaultPaths(base_dir)
paths.cache_dir          # base_dir/.knox/cache
paths.compile_state_path # base_dir/.knox/compile_state.json
paths.wiki_cache_path    # base_dir/.knox/last_wiki.json
paths.extract_dir        # base_dir/.knox/cache/extract
paths.images_dir         # base_dir/.knox/cache/images
```

**Usage patterns:**

| Context | Pattern |
|---------|---------|
| Module has `config: PipelineConfig` | `config.paths.cache_dir` (uses `PipelineConfig.paths` property) |
| Module has `base_dir: Path` | `VaultPaths(base_dir).some_path` |
| Module has `output_dir: Path` (= base_dir/wiki) | `VaultPaths(output_dir.parent).some_path` |
| Module has `state_dir` (= base_dir/.knox) | Direct `state_dir / CONSTANT` (no VaultPaths needed) |

**Rule:** Never construct `.knox/` paths by hand from `output_dir` (e.g.
`output_dir / ".knox" / ...`). This was the source of multiple path bugs
where artifacts landed in `wiki/.knox/` instead of `.knox/`. Always use
`VaultPaths(base_dir)` or `VaultPaths(output_dir.parent)`.

**Tests** must also import from `knox.core.paths` — never hard-code path
strings in test assertions.

- All tests use `pytest` and `tmp_path` for filesystem operations.
- Unit tests must mock external dependencies (LLM, network, filesystem).
- Integration tests that call LLM APIs must use `@pytest.mark.skipif`
  checking for API key availability.
- Test files follow `test_<module>.py` naming.

### Issue Tracking

This project uses **bd (beads)** for all task tracking. Load the `beads` skill
for workflow details, commands, and session close protocol. Do NOT use
TodoWrite, TaskCreate, or markdown TODO lists — use `bd` exclusively.

---

## Verification

- Run `ruff check` on changed Python files before committing
- Run `uv run pytest tests/unit -x --timeout=30` for unit test coverage
- Run `uv run pytest tests/integration -x --timeout=120` for integration tests
- Run `ty check src/knox` for type checking (mypy is not used)
- Run `uv run python scripts/check_field_consistency.py` to catch hard-coded contract keys
- Run `uv run python scripts/check_antipatterns.py` to catch banned-noqa, magic crops, `try/except ImportError` guards, and `global` keyword usage
- Verify `knox.toml.example` stays in sync with `DEFAULTS` when adding config keys
- After changing `src/knox/core/paths.py`, verify no hard-coded path strings
  exist in other `.py` files
- After changing CLI commands, MCP tools, config keys, or output format, check
  `skills/knox/SKILL.md` for stale or missing info
- Run `uv run mkdocs build --strict` when changing documentation files

## New Code Checklist

Tick these BEFORE writing a function. If you can't, stop and fix the
foundation first — the alternatives are all antipatterns:

- **Limit value** (number of items, max chars, max tokens) — does it come
  from `config.get(...)` / `DEFAULTS.get(...)` / a `PipelineConfig`
  field? If you typed a literal like `[:10]` or `text[:200]`, you're
  about to silently drop data. Use the `DisplayConfig` section for
  display limits or `BudgetConfig` for LLM input/output budgets.
- **Role / task / compiler / tier value** — does it come from a
  `StrEnum` member (`LLMRole`, `AgentTask`, `Compiler`, `Tier`)?
  If you typed a bare string like `role="compile.extract"`, the
  field-consistency scanner will reject the commit.
- **Import** — is it at module level? If you typed
  `from somewhere import x` inside a function body, the antipattern
  scanner will reject the commit. The only legitimate exception is
  a `try/except ImportError` probe for an **explicitly optional**
  package declared in `pyproject.toml`'s `[project.optional-dependencies]`.
- **Exception truncation** — is the limit coming from
  `shorten_exception(exc)` (which uses the `MAX_CHARS_EXCEPTION`
  constant)? If you typed `str(exc)[:200]`, you're hard-coding the
  limit; the helper exists precisely for this.
- **Path construction** — are you using `pathlib.Path` with `/` for
  joins? If you typed `+ "/" +` for path concatenation, stop and use
  the proper operator.
- **Mutable default** — is the default a factory (`field(default_factory=...)`)
  or a small literal? If it's a mutable container (`list`, `dict`, `set`),
  you have a shared-state bug waiting to happen.
- **Noqa suppression** — is `# noqa: RULE` actually necessary? If you
  can't justify it in a comment, the right fix is to refactor the
  code, not suppress the warning. `PLC0415` and other banned-noqa
  rules cannot be suppressed at all.

### Enforcement scripts

| Script | Catches | Lives in |
|---|---|---|
| `ruff check` | Most lint issues, but PLC0415/E402 are still warnings (not errors) | ruff.toml |
| `scripts/check_field_consistency.py` | Bare-string contract keys (field names, role=, task=, compiler=, tier=) | scripts/ |
| `scripts/check_antipatterns.py` | Banned-noqa, magic crops, `try/except ImportError` guards, `global` keyword | scripts/ |

All three are run in CI; the last two exit non-zero on any violation.
Run them locally before pushing: `mise run check-all` (or the
individual commands above).

### Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

### User Preferences

- The user prefers concise, direct answers with no filler or intros
- Use `uv run` for all Python execution, never direct `python` calls
- All task tracking via `bd` (beads) — no markdown TODO lists
- English-only for code, docs, and commits
- Conventional commits format (`feat:`, `fix:`, `chore:`, `docs:`, etc.)

---

### Child DOX Index

| Path | Scope |
|------|-------|
| `src/knox/AGENTS.md` | Python source tree — text-in-code conventions, antipatterns, Jinja2 templates, pack structure |
| `tests/AGENTS.md` | Testing — unit, integration, benchmarks, fixtures, helpers, test antipatterns |
