# DOX — kompend Python source tree

<!-- on entry: loaded when agent enters src/kompend/ -->
<!-- parent: ../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Coding conventions and quality standards for the kompend Python source tree at
`src/kompend/`. Governs how Python code is written and how human-readable text is
handled. Code policies (antipatterns, boundaries, error handling) are universal
and live in `.agents/CODE.md`.

## Ownership

- `src/kompend/` — all kompend Python source code
- `src/kompend/templates/` — all LLM prompt and instruction templates
- `src/kompend/packs/` — prompt packs (see `packs/AGENTS.md`)

## Local Contracts

### No text in code

**LLM prompts, system instructions, template strings, configuration data, and
any other human-readable text must NOT be embedded in Python source files.**
Text in string constants is invisible to the domain experts who review prompts,
editable only by touching Python, versioned with code that has a different
review cycle, and unverifiable by tooling.

| Content | Format | Location |
| ------- | ------ | -------- |
| LLM system/user prompts | Jinja2 (`.md.j2`) | `templates/<module>/<role>_system.md.j2` |
| Page kind definitions | TOML | `<module>/kinds.toml` or config |
| Pack prompts and roles | TOML | `packs/<name>/pack.toml` |
| Any other human-readable string | TOML / Jinja2 | the appropriate config dir |

**Acceptable exceptions:** short (≤5 words) status messages to `typer.echo()` or
a logger; error messages that must name code-level constructs (variables,
types); single words as enum members, dict keys, or field names; docstrings and
type annotations.

Template layout, loader search order, frontmatter schema, and variable
validation — see `core/templates/AGENTS.md`. Pack layout — see
`packs/AGENTS.md`. Known legacy violations are tracked in beads (`bd list`); the
ban's rationale — see `.agents/history/2026-10-23-rationale-archive.md`.

### Python antipatterns

Antipattern table, code boundaries (config fields, no absolute paths, no
`TYPE_CHECKING`, banned suppressions, lazy heavy deps), and the error-handling
policy — **see `.agents/CODE.md`**. They apply to production code and tests
alike. Two are enforced by script; run both before committing (see
`.agents/CHECKS.md`).

**Never crop.** No `text[:N]`, no arbitrary char/token cap on a prompt, a
source, a model answer or a result set, and no `max_tokens` used to shorten an
output. A budget shapes the work by splitting it into windows; it never
shortens or drops it (root AGENTS.md, `.agents/CODE.md`).

### Output format — OKF v0.2

kompend emits **OKF v0.2** bundles by default; OKF is a compliance spec, not an
output format. Conformance rules (frontmatter, trust fields, provenance fields,
portable cross-links, reserved filenames, page kinds, footnote attribution,
permissive consumption) — **see `output/AGENTS.md`** and
`docs/concepts/open-knowledge-format.md`. Generated output is rumdl-clean: there
is one markdown writer, no exempt variants.

### LLM agents

Every LLM call routes through the PydanticAI agents in `agents/` — class
hierarchy, `KompendAgentBase` responsibilities, `KompendModel`, deps types, and the
rules for adding an agent — **see `agents/AGENTS.md`**. Model construction goes
through `kompend.llm.models.build_model()`; see `llm/AGENTS.md`.

### ACP agent

The Agent Client Protocol adapter (`kompend agent` on stdio) — protocol flow,
module map, and modification guidance — **see `acp/AGENTS.md`** and
`docs/integrations/acp-agent.md`.

## Work Guidance

- Python 3.13+ only. Modern typing (`X | None`, `list[str]`).
- Every module in `src/kompend/` has an `__init__.py` that defines its public API
  via re-exports. Import from the package root, not the defining submodule.
- `src/kompend/core/paths.py` is the single source of truth for on-disk artifact
  paths — see `.agents/FILES.md`.
- `src/kompend/core/config/` owns all configuration and `DEFAULTS` — see
  `core/config/AGENTS.md`.
- Heavy optional dependencies stay lazy so CLI startup stays fast — pattern in
  `.agents/CODE.md`.

## Verification

Commands — see `.agents/CHECKS.md`. For this subtree the relevant ones are
`ruff check`, `ty check src/ tests/`, `pytest tests/unit`, and the two AST gates.
`mise lint` and `mise typecheck` wrap the first two.

## Child DOX Index

| Path | Scope |
| ---- | ----- |
| `acp/` | ACP adapter — TOML-defined conversational agents, tool set, provider store |
| `agents/` | PydanticAI agents for every LLM call; `KompendAgentBase`, `KompendModel`, deps |
| `capture/` | Capture/distill/learn pipeline for URLs, files, stdin |
| `chunking/` | Tier 2 — split parsed documents into context-window-sized `ChunkTree` |
| `cli/` | CLI shell over the core library — Typer commands, argument declarations |
| `cli/args/` | All CLI argument/option definitions (`Annotated` aliases) |
| `cli/cmds/` | All CLI command entry points, organized by domain |
| `compilation/` | Tier 4 — knowledge graph build, WiCER loop, page kinds |
| `core/` | Foundation — models, pipeline, config, cache, templates, trust, utils |
| `core/cache/` | Content-addressed JSON cache with dependency validation |
| `core/config/` | Configuration resolution engine and `DEFAULTS` |
| `core/models/` | Canonical data models and serialization contracts |
| `core/pipeline/` | Tier orchestration, pipeline config, tier state, ingest |
| `core/reporting/` | On-demand report builders for CLI observability |
| `core/resolution/` | Tier 3 — entity extraction/dedup, cross-reference checking |
| `core/store/` | The vault SQLite store (`.kompend/kompend.db`) |
| `core/templates/` | Jinja2 loading, frontmatter parsing, variable validation |
| `core/trust/` | Actor identity and trust/privacy concerns |
| `core/utils/` | Pure helpers, zero domain logic |
| `lint/` | Structural + LLM lint rules, gap analysis, remediation |
| `llm/` | Model construction, context-window discovery, formats, usage tracking |
| `mcp/` | MCP server exposing kompend operations as tools (fastmcp) |
| `memory/` | Query memory store and compile-time lifecycle |
| `output/` | Tier 5 — OKF markdown/JSON/Typst writers, indexes, log |
| `preprocessing/parsers/` | Pluggable ingestion backends and their registry |
| `rag/` | zvec vector store — fields, index build, filters, drift, pruning |
| `search/` | Multi-retriever semantic search with adaptive selection |
