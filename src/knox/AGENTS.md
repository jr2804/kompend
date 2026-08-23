# DOX — knox Python source tree

<!-- on entry: loaded when agent enters src/knox/ -->
<!-- parent: ../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Coding conventions and quality standards for the KNOX Python source tree at
`src/knox/`. This doc governs how Python code is written, how LLM prompts and
human-readable text are handled, and what patterns are forbidden.

## Ownership

- `src/knox/` — all KNOX Python source code
- Templates, config data, and any human-readable text within the source tree
- Packs (prompt bundles) in `src/knox/packs/`

## Local Contracts

### No text in code

**LLM prompts, system instructions, template strings, configuration data, and
any other human-readable text must NOT be embedded in Python source files.**

This rule exists because:

- **Discoverability** — Text hidden in Python string constants is invisible to
  non-developers (domain experts, technical writers) who need to review prompts.
- **Editability** — Changing a prompt requires editing a Python file, which
  risks introducing syntax errors, import issues, or accidental code changes.
- **Versioning** — Prompts and code have different review cycles. A prompt tweak
  should not recompile the Python package.
- **Tooling** — Jinja2 templates can be syntax-checked, diffed, and previewed.
  TOML config can be validated with a schema. Python string constants cannot.

#### What goes where

| Content                         | Format            | Location                                 |
| ------------------------------- | ----------------- | ---------------------------------------- |
| LLM system prompts              | Jinja2 (`.md.j2`) | `<module>/templates/<role>_system.md.j2` |
| LLM user prompts / templates    | Jinja2 (`.md.j2`) | `<module>/templates/<role>_prompt.md.j2` |
| Page kind definitions (N9)      | TOML              | `<module>/kinds.toml` or config          |
| Pack prompts and roles          | TOML              | `packs/<name>/pack.toml`                 |
| Any other human-readable string | TOML / Jinja2     | appropriate config dir                   |

#### How templates are loaded

All templates go through `knox.core.templates.TemplateLoader`, which
searches three locations in order (project > user > built-in):

| Priority    | Location              | Example path                               |
| ----------- | --------------------- | ------------------------------------------ |
| 1 (highest) | Project overrides     | `./knox/templates/` (next to `knox.toml`)  |
| 2           | User global overrides | `~/.config/knox/templates/`                |
| 3 (lowest)  | Built-in (package)    | `importlib.resources` in `knox/templates/` |

Usage:

    from knox.core.templates import get_template_loader
    loader = get_template_loader()
    rendered = loader.render("lint/clarity-score.md.j2",
                             title="My Page", content="...")

#### Template variable validation

Every template is validated at render time:

- **StrictUndefined** — Any `{{ var }}` referenced in the template must have
  a matching kwarg. Missing variables raise `jinja2.UndefinedError` immediately.
- **Required variable check** — If the template declares `required: true` in
  its frontmatter, the loader raises `ValueError` before rendering if the
  variable is missing.
- **Unused variable warning** — Kwargs never referenced in the template are
  logged at `WARNING` level. Catches typos like `contnt` vs `content`.
- **AST introspection** — Uses `jinja2.meta.find_undeclared_variables()` to
  determine which variables the template actually uses, without rendering it.

#### Acceptable exceptions

- Short (≤5 words) status messages passed to `typer.echo()` or logger calls
- Error messages that must reference code-level constructs (variable names, types)
- Single words used as enum members, dict keys, or field names
- Docstrings and type annotations

#### Known violations

See `.agents/HISTORY.md` for the T1–T10 text-in-code violations table and
historical design decisions.

### Python Antipatterns

1. **Follow the `python-ultimate` skill.** When in doubt about Python coding
   style, structure, or conventions, load and follow the guidelines in the
   `python-ultimate` skill. It takes precedence over any conflicting habit
   or default.

2. **Universal rules — see `.agents/POLICIES.md`:**
   - Magic numbers/hardcoded values → config fields (Antipatterns 1-4)
   - Error handling — "let it burn" (propagate, don't mute)
   - Absolute paths in `.knox/` artifacts (Antipattern 6)
   - No inline imports (PLC0415), no `TYPE_CHECKING`, no `try/except ImportError`

3. **New code checklist — see `.agents/CHECKS.md`:**
   - Limit values from config, not literals
   - Enum values from StrEnum, not bare strings
   - Imports at module level
   - No global state
   - Noqa suppressions with justification

### Antipattern summary

| Pattern | Verdict | Fix |
|---|---|---|
| Hard-coded model name (`model="ollama/llama3.2"`) | ❌ | Config field default, never a function signature |
| Domain value as function default (`prompt_budget: int = 200_000`) | ❌ | Config dataclass field |
| `x: T | None = None` with hidden default in body | ❌ | Config dataclass field; docstring if None truly needed |
| `content[:200]` / `text[:4096]` | ❌ | Full content, or batch with a config/model-derived limit |
| Config fallback that must match a runtime provider's output (`dim = 384`) | ❌ | Read the value from the provider at runtime (`provider.dim`), never a config literal |
| Bare schema/field-name literals (`hit.get("source_file")`) | ❌ | `StrEnum` / `Final[str]` symbol imported at every site |
| Hard-coded `"KNOX_*"` env-var literal in env access (`os.environ.get("KNOX_WIKI_DIR")`, `envvar=`) | ❌ | `EnvVar` member from `knox.core.env` (single definition; enforced by `scripts/checks/check_antipatterns.py` `ENV-VAR-LITERAL`) |
| Stable-shape dict without a schema (JSONL records, persisted entries, spec payloads: lint history, dedup store, OKF sources, graph edges, citations) | ❌ | `TypedDict` next to the owning model — types keys with zero runtime/on-disk cost. Dataclasses only where the shape is also constructed/validated in code, not just serialized |
| Absolute path in a `.knox/` artifact (`citing_source = str(path)`) | ❌ | `rel_to_vault(path, base_dir)` at the persistence boundary |

#### Jinja2 conventions

- Template files use the `.md.j2` extension (markdown + Jinja2).
- Variables use `{{ snake_case_name }}` syntax.
- Control flow uses `{% for %}` / `{% if %}` blocks, never Python string
  concatenation.
- Each template file **starts with a YAML frontmatter block** delimited by
  `---`. This frontmatter is parsed by `TemplateLoader` and does not appear in
  the rendered output:

---

variables: - name: title
required: true - name: content
required: true - name: context_budget
required: false
role: lint.factual-consistency
response_format: json_object

### temperature: 0.0

    ---

The frontmatter fields are:

- `variables` — list of expected variables with `name` and `required`.
- `role` — pipeline role (e.g. `"lint.clarity-score"`).
- `response_format` — LLM response format (`"json_object"` or `"text"`).
- `temperature` — LLM temperature override for this specific template.

Required variables declared in frontmatter are checked by the loader. If a
required variable is missing, `ValueError` is raised before the template is
ever rendered.

#### Adding new text — checklist

Before adding a string constant to a Python file, ask:

1. **Is this human-readable text?** If yes, it belongs in a template or config file.
2. **Is this an LLM prompt?** → Jinja2 template in `<module>/templates/`.
3. **Is this configuration data?** → TOML in `<module>/` or the global config.
4. **Is this a short status message?** → Acceptable in Python, but keep it ≤5 words.
5. **Is this an error message referencing code internals?** → Acceptable in Python.
6. **Have I added a `templates/` directory if one doesn't exist?** → Do it.

#### Pack structure

When moving prompts out of `packs/builtins/<name>.py`, use this layout:

- `src/knox/packs/<name>/pack.toml` — manifest + prompts (see TOML schema)
- `src/knox/packs/<name>/__init__.py` — optional Python registration
- `src/knox/packs/<name>/templates/` — optional Jinja2 templates
  (e.g. `compile.md.j2`)

The `pack.toml` manifest:

    [pack]
    name = "academic"
    description = "Prompts for academic research papers"
    version = "1.0.0"

    [pack.dependencies]
    # other pack names

    [roles.compile]
    template = "compile.md.j2"
    response_format = "json_object"
    temperature = 0.0

    [roles.resolve]
    template = "resolve.md.j2"
    response_format = "json_object"
    temperature = 0.0

### OKF compliance — output format rules

KNOX produces **OKF v0.2** (Open Knowledge Format) bundles by default. OKF is
NOT an output format — it is a **compliance spec** for the markdown output.
The spec is at <https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md>

`compile.output.format` offers only real formats: `markdown` (default, OKF
v0.2-compliant), `json`, `typst`.

**All output-related code must maintain OKF v0.2 conformance:**

1. **Frontmatter** is built once by `VaultWriter.okf_frontmatter()` — every
   vault writer MUST delegate to it (single source of truth). Every concept
   file has parseable YAML frontmatter with a non-empty `type` field (§11.2).
2. **Trust fields:** `generated: {by, at}` (§5.2; `by` uses the §7 actor
   convention `knox/<version>`) supersedes the v0.1 `timestamp`. `verified`
   (§5.2) is emitted only when a verification pass ran; `status` defaults to
   `stable` (§5.4).
3. **Provenance fields are distinct:** `tags` = topic tags only (no filename
   overload); `source_files` = KNOX extension (provenance filename stems);
   `sources` = OKF §5.1 structured objects `{id, resource, title, author}` from
   the reference store — NEVER bare strings.
4. **Cross-links** are portable, folder-relative markdown paths:
   `[Title](concepts/title.md)` from the bundle root, `[Title](title.md)` from
   inside `concepts/` — never `[[wikilinks]]`, never leading-slash absolute
   links. Portable Markdown must render outside Obsidian too.
5. **Reserved filenames** (`index.md`, `log.md`) follow OKF structure (§8, §9).
   Root `index.md` MAY declare `okf_version: "0.2"` in frontmatter (§12).
   `log.md` follows §9: date-grouped `## YYYY-MM-DD` headings, newest first,
   entries as `- [HH:mm:ss] **Action**: prose` list items; it is never
   wiped — compiles, ingests, queries, and lint runs all append entries to
   the same file (`log_stage` in `output/pages/log.py`).
6. **Page kinds** follow the mandatory taxonomy (`concept`, `entity`,
   `summary`, `comparison` — `MANDATORY_KINDS` in `core/config/keys.py`)
   plus `compile.extraction.categories` extras; the LLM assigns `kind` per
   page, unknown values fall back to `concept` (see `compilation/AGENTS.md`).
   Extracted entities become mandatory generated pages (`kind: entity`,
   routed to `entities/`) unless a concept page with the same key exists.
7. **Per-claim attribution** uses markdown footnotes `[^id]` keyed to
   `sources[].id` (§5.1) — NOT a v0.1 body `# Citations` numbered list. Inline
   `[sources/x.pdf]` citations are rewritten to `[^id]` refs; definitions are
   emitted only for referenced ids, ordered by first reference.
8. **Consumers MUST tolerate** missing optional fields, unknown `type` values,
   unknown frontmatter keys, broken cross-links, and missing `index.md` files
   (§10 — permissive consumption model).

**Generated output must be rumdl-clean**: compile runs `rumdl fmt` over the
output and lint uses the vault's rumdl config (`compile.output.rumdl_config`,
default `.knox/rumdl.toml`, auto-materialized from the shipped default).
There is a single markdown writer (OKF-compliant); no exempt variants.

### ACP agent — credential-free LLM injection and conversational agents

KNOX implements the **Agent Client Protocol (ACP)** to let editors and AI
agents inject their LLM endpoint without sharing credentials. The agent runs
as `knox agent` on stdio.

**Protocol flow:**

1. Client calls `initialize` \u2192 KNOX advertises `providers: {}` capability
2. Client calls `providers/list` \u2192 KNOX returns current LLM endpoint info
3. Client calls `providers/set` \u2192 KNOX stores the endpoint and reconfigures the model
4. Client calls `session/new` \u2192 KNOX creates a session
5. Client sends `/knox <command>` in a prompt \u2192 KNOX runs the pipeline

**Two prompt paths:**

- **Conversational agent** (default when LLM is configured): prompts go through
  a tool-calling loop driven by TOML-defined agents (`builtin/*.toml` +
  `.knox/agents/*.toml`). The LLM reasons over available tools and calls
  them as needed. See `knox agents list` for available agents.
- **Subprocess dispatch** (degraded mode, no LLM): `/knox` slash commands
  launch the corresponding CLI subcommand. Free-text prompts are rejected.

**Key modules:**

- `knox_agent.py` — ACP protocol handler, routes prompts to conversational or subprocess path
- `definitions.py` — `AgentDefinition` pydantic model loaded from TOML
- `registry.py` — `AgentRegistry` discovers agents from builtin + vault
- `tools.py` — `ToolDispatcher` wraps existing MCP server functions as agent-callable tools
- `conversational.py` — `ConversationalAgent` tool-calling loop
- `provider_store.py` — thread-safe LLM credential store for providers/set
- `slash_commands.py` — slash command parsing and structured definitions

**Key design decisions:**

- **KNOX controls its own LLM calls.** Unlike MCP sampling (free-text only),
  KNOX uses its full pydantic-ai agent + structured output pipeline. The
  `providers/set` endpoint configures the LLM; KNOX makes direct HTTP calls.
- **`providers/set` is an RFD** (not yet in the stable ACP spec). The Python
  SDK v0.11.0 includes the types. When `providers/set` is not called, KNOX
  falls back to env vars \u2192 config.toml \u2192 DEFAULTS (unchanged).
- **Slash commands** (`/knox compile`, `/knox query`, etc.) are parsed from
  the prompt text and dispatched via subprocess. The agent runs on asyncio.
- **Sessions are stateless.** Each prompt turn runs independently. No
  conversation history is accumulated between turns.
- **MCP companion mode is v2.** The existing `knox mcp` subcommand is
  unchanged. The ACP agent does not bridge to MCP in v1.

**When modifying the agent module (`src/knox/acp/`):**

- The `KnoxAgent` class implements the `acp.Agent` protocol. All protocol
  methods are async. Use `asyncio.run()` to bridge to synchronous code.
- `ProviderCredentials` is thread-safe (`threading.Lock`). It reconfigures
  the model via env vars (`KNOX_LLM_DEFAULT_MODEL`, `KNOX_LLM_API_BASE`,
  `KNOX_LLM_API_KEY`) and restores them on `disable()`.
- Slash commands are parsed by `parse_slash_command()` in
  `slash_commands.py`. Add new commands there.
- The `ext_method()` handler on `KnoxAgent` receives unknown ACP methods
  (including `providers/*`). Add new provider methods there.

### PydanticAI pipeline agents (`src/knox/agents/`)

All LLM stages in the compilation pipeline now route through PydanticAI
agents instead of direct LLM client calls. Seven agents cover
the pipeline stages:

| Agent                   | Stage                      | Output                       |
| ----------------------- | -------------------------- | ---------------------------- |
| `ExtractionAgent`       | TwoPhase extraction        | `ExtractionOutput` (JSON)    |
| `ExtractionReduceAgent` | TwoPhase extraction reduce | `ExtractionOutput` (JSON)    |
| `GenerationAgent`       | TwoPhase generation        | `GenerationOutput` (JSON)    |
| `QueryAgent`            | RAG query synthesis        | `str` (free text)            |
| `WicerProbeAgent`       | WiCER probe generation     | `WicerProbeOutput` (JSON)    |
| `WicerEvaluateAgent`    | WiCER evaluation           | `WicerEvaluateOutput` (JSON) |
| `WicerDiagnoseAgent`    | WiCER diagnosis            | `WicerDiagnoseOutput` (JSON) |
| `VlmAgent`              | VLM image captioning       | `str` (caption)              |

**Key design:**

- Agents use native pydantic-ai models via
  :func:`knox.llm.models.build_model`, wrapped in :class:`KnoxModel`
  (usage tracking + debug saving) and :class:`FallbackModel` (failover).
  Custom providers (`claudinio/`, `ollama-cloud/`, `opencode-zen/`,
  `opencode-go/`) construct native models pointed at the provider's
  endpoint; standard prefixes (`ollama/`, `openai/`, `anthropic/`,
  etc.) resolve via pydantic-ai's `infer_model()`.
- `KnoxAgentBase` handles lazy agent construction, template rendering,
  skill injection, and error wrapping. `_build_agent()` is concrete —
  subclasses only declare class vars (`template_name`,
  `system_template_name`, `result_type`, `deps_type`).
- Structured output uses ToolOutput mode (default) — tool-call arguments
  are validated directly by pydantic-ai, so no JSON-text salvage is
  needed.

**When modifying agents (`src/knox/agents/`):**

- Add new agents in `src/knox/agents/<name>.py`, following the existing
  pattern (subclass `KnoxAgentBase`, set `result_type` and
  `template_name`, implement `_build_agent()`).
- Model strings use the `provider/model` convention (split at first
  `/`); `build_model()` translates to native pydantic-ai models.
  Never construct pydantic-ai models directly in business code.
- `KnoxModel` (`knox_model.py`) wraps any native delegate, adding
  usage tracking and debug saving. `FallbackModel` wraps primary +
  fallback `KnoxModel` instances for failover.
- `KnoxAgentBase.run()` accepts an optional `message_history` kwarg
  for multi-turn conversations (used by incremental extraction).
  `run_with_messages()` returns `(output, new_messages)` for callers
  that need to thread conversation history across calls.
- Template variables are filtered to only include those declared in the
  template's frontmatter. Add new variables to the template's
  `variables:` list.
- `debug.py` holds `save_debug_prompt(...)` — the per-LLM-call dump
  triggered by `KNOX_DEBUG_LLM_DIR`. `KnoxModel` calls it after each
  request; `VlmAgent.run()` sets `current_source_id` then calls it.

### Work Guidance

- Python 3.11+ only. Use modern typing syntax (`X | None`, `list[str]`).
- Every module in `src/knox/` must have an `__init__.py` (even if empty) that
  defines the public API via re-exports.
- `src/knox/core/paths.py` is the single source of truth for all on-disk
  artifact path constants.
- `src/knox/core/config/` governs all configuration — see its AGENTS.md for
  config-specific rules.

### Verification

- `ruff check src/knox/` — must pass without errors
- `ty check src/knox/` — type checking (mypy is not used)
- `uv run pytest tests/unit/ -x --timeout=30` — unit tests pass
- `uv run pytest tests/integration/ -x --timeout=120` — integration tests pass

---

### Child DOX Index

| Path                              | Scope                                                                                                                                                                                                                                             |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `acp/`                            | ACP agent — `providers/set` credential injection, TOML-defined conversational agents, tool-calling loop, slash commands                                                                                                                           |
| `agents/`                         | PydanticAI pipeline agents — `ExtractionAgent`, `GenerationAgent`, `QueryAgent`, `WicerProbeAgent`, `WicerEvaluateAgent`, `WicerDiagnoseAgent`. All LLM stages route through native pydantic-ai models via `build_model()` + `KnoxModel` wrapper. |
| `cli/AGENTS.md`                   | CLI layer — thin shell over core, Typer commands, argument declarations                                                                                                                                                                           |
| `core/AGENTS.md`                  | Core foundation — models, paths, config, pipeline, cache, templates, trust, reporting, utils, resolution                                                                                                                                          |
| `llm/AGENTS.md`                   | LLM client — error hierarchy, failover (provider rules delegated to `providers/AGENTS.md`)                                                                                                                                                        |
| `memory/AGENTS.md`                | Memory layer — query memory store + compile-time lifecycle                                                                                                                                                                                        |
| `preprocessing/parsers/AGENTS.md` | Pluggable ingestion backends — registration, ParserBuildOptions, figure_enrichment semantics                                                                                                                                                      |
