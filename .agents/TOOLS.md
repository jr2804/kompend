# AGENTS.md — TOOLS

State file: inventory of development-time tooling — CLIs, MCP servers, skills,
and project tasks. Demand-loaded; keep it current (see `.agents/MAINTENANCE.md`).

## Rules

- Use an existing tool before adding a new one. No new tool, MCP server, skill,
  or dependency without explicit instruction (`.agents/POLICIES.md`).
- Run `<tool> --help` before the first use in a session — never guess a flag.
- Each tool appears once, in its primary table. Delete rows when tools leave;
  this is not a historical log.

## CLI tools & runtimes

| Tool | When | Purpose |
| ---- | ---- | ------- |
| `uv` | Always | Venv and dependency manager; every Python call is `uv run …`, never `python`/`pip` |
| `mise` | Running a repo task | Task runner and tool pinning; tasks in `.config/mise/conf.d/` |
| `ruff` | Lint or format Python | `ruff check`, `ruff format` |
| `ty` | Type check | The project type checker — not mypy or pyright |
| `rumdl` | Markdown lint/format | Config `.config/rumdl.toml`; also called from code by the markdown writers |
| `rord` (pyreorder) | Structural Python ordering | Imports → constants → classes → functions; config `.config/reorder.toml` |
| `codespell` | Before commit | Spelling over `src/ tests/ scripts/` |
| `jactionlint`, `shellcheck` | Editing `.github/workflows/` | Workflow linter; shells out to shellcheck for `run:` blocks |
| `pytest` | Any code change | Test runner; conventions in `tests/AGENTS.md` |
| `mkdocs` | Docs change | `uv run python -m mkdocs build --strict` — module form, the console shim fails on Windows |
| `bd` (beads) | Task tracking | `bd ready`, `bd show <id>`, `bd update <id> --claim`, `bd close <id>`, `bd prime` for full context |
| `dolt` | Beads storage/sync | Backing store for the beads issue database |
| `codegraph` | Navigating unfamiliar code | Symbol search, call edges, impact analysis |
| `grepai` | Finding code by intent | Semantic search — meaning, not text |
| `jev` | Judging text | Typed judgments: screening, ranking, claim verification |
| `ketch` | External research | Web search, OSS code search, library docs, scrape, crawl |
| `rg`, `tree-cli` | Any tree request | `rg --files \| tree-cli --fromfile` — never hand-draw a tree |
| `git`, `gh` | History and remote ops | `gh auth switch -u jr2804` must share the command with the push — the switch does not persist |

## MCP servers

Project-registered servers are driven by `.config/mise/data/dev-features.csv`;
community skills by `.config/mise/data/skills.csv`. Both are the single source of
truth and start with an `enabled` column. `mise run add-mcp-servers <agent>` and
`mise run add-skills` install `enabled=1` rows and *remove* `enabled=0` rows
(tool via `mise remove --env local`, server via `add-mcp remove`, skill via
`skills remove`) — a dormant row is an uninstall instruction, not a no-op.
The `<agent>` names follow `add-mcp` (claude-code, pi, opencode, ...); the
script's `AGENT_ALIASES` maps the friendly spellings claude/gemini/copilot onto
them.
Dev-feature tools are pinned by those tasks into the gitignored `mise.local.toml`,
so `.config/mise/config.toml` carries no per-checkout tool state.

| Server | When | Purpose |
| ------ | ---- | ------- |
| `beads` | Task tracking over MCP | Same database as `bd` |
| `codegraph` | Code navigation | Knowledge-graph queries |
| `grepai` | Semantic code search | Meaning-based retrieval |
| `jev` | Judgments | Screen, rank, verify |
| `ketch` | Research | Search, docs, scrape, crawl |
| `aislop` | Slop audit | AI-generated-code pattern detector |
| `fartrun` | Health/security sweep | Dead code, tech debt, secrets |
| `repowise` | Architecture and health | Dependency graph, docs, decisions |
| `vibecheck` | AI code smells | JS/TS/Python smell scan |
| `hound`, `cytoscnpy` | Runtime-provided | No project skill documents them — read the server's own tool description before use |

## Skills

Installed under `.agents/skills/<name>/SKILL.md` (43 skills) — load the file,
don't guess its rules.

| Group | Skills | When |
| ----- | ------ | ---- |
| Code navigation | `cli-codegraph`, `cli-grepai`, `codebase-exploration`, `architectural-decisions`, `code-health`, `dead-code-cleanup` | Before reading unfamiliar code |
| Quality sweeps | `cli-aislop`, `cli-ai-slop-detector`, `cli-sloppylint`, `cli-vibecheck`, `cli-fartrun`, `cli-deep-slop-scan` | Auditing AI-generated code |
| Python craft | `python-ultimate`, `solid`, `debugging-code`, `pyreorder`, `uv` | Writing, reviewing, or debugging Python |
| Docs and prose | `project-docs`, `good-readme`, `stop-slop`, `beads-docs`, `agent-rules` | Documentation, README, AGENTS.md work |
| Visual deliverables | `html`, `html-diagram`, `html-plan`, `html-prototype`, `html-wireframe`, `design-artifact`, `visual-explainer`, `mermaid-diagrams`, `svg-precision-skill`, `scientific-figures`, `beautify-github-readme` | Any figure, diagram, or page artifact |
| Repository operations | `beads`, `releasing`, `improve`, `add-lang`, `agent-eval`, `code-deduplication`, `jev`, `ketch`, `rtk-optimizer` | Issue lifecycle, releases, audits, capability additions |

## Project tasks & scripts

Run through `mise run <task>`.

| Command | Purpose |
| ------- | ------- |
| `mise run test` | pytest with coverage |
| `mise run lint` | ruff check with fixes |
| `mise run typecheck` | `ty check src/ tests/ scripts/` |
| `mise run format`, `mise run format-md` | pyreorder + ruff; rumdl for markdown |
| `mise run spell` | codespell |
| `mise run lint-actions` | jactionlint over all workflows |
| `mise run docs` | mkdocs strict build |
| `mise run all` | Every quality gate above, in order |
| `mise run check-all` | AST antipattern checkers (see below) |
| `mise run add-mcp-servers <agent>`, `mise run add-skills` | Reconcile dev-feature tools, MCP servers, skills from the CSVs |
| `scripts/checks/check_antipatterns.py` | Banned noqa, magic crops, `try/except ImportError`, `global` |
| `scripts/checks/check_field_consistency.py` | Bare-string literals used as cross-file contract keys |
| `scripts/checks/check_dox.py` | DOX size budgets and near-duplicate sentences |
| `scripts/checks/check_cli_surface.py` | CLI command surface contract |
| `scripts/checks/check_config_literals.py` | Config keys resolved through `DEFAULTS`, not literals |
| `scripts/checks/check_persisted_paths.py` | Artifact names come from `paths.py` |
| `scripts/checks/check_enum_cache.py` | Enum cache invariants |
| `scripts/checks/check_mcp_parity.py` | Every CLI command has its MCP tool |
