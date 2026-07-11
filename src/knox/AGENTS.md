# DOX — knox Python source tree

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

| Content | Format | Location |
|---------|--------|----------|
| LLM system prompts | Jinja2 (`.md.j2`) | `<module>/templates/<role>_system.md.j2` |
| LLM user prompts / templates | Jinja2 (`.md.j2`) | `<module>/templates/<role>_prompt.md.j2` |
| Page kind definitions (N9) | TOML | `<module>/kinds.toml` or config |
| Pack prompts and roles | TOML | `packs/<name>/pack.toml` |
| Any other human-readable string | TOML / Jinja2 | appropriate config dir |

#### How templates are loaded

All templates go through `knox.core.template_loader.TemplateLoader`, which
searches three locations in order (project > user > built-in):

| Priority | Location | Example path |
|----------|----------|-------------|
| 1 (highest) | Project overrides | `./knox/templates/` (next to `knox.toml`) |
| 2 | User global overrides | `~/.config/knox/templates/` |
| 3 (lowest) | Built-in (package) | `importlib.resources` in `knox/templates/` |

Usage:

    from knox.core.template_loader import get_template_loader
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

#### Known violations (T1–T10)

These are filed as beads issues and must be fixed before adding new text-in-code:

| Issue | File | What |
|-------|------|------|
| `knox-ox2` (T1) | `lint/llm_rules.py` | 3 LLM prompt strings for lint rules |
| `knox-8b0` (T2) | `packs/base.py` | 4 default pack prompt strings |
| `knox-1a8` (T3) | `packs/builtins/academic.py` | 3 academic pack prompt strings |
| `knox-65z` (T4) | `packs/builtins/technical.py` | 3 technical pack prompt strings |
| `knox-174` (T5) | `compilation/schema.py` | BUILTIN_KINDS dict with descriptions |
| `knox-dnu` (T6) | `compilation/two_phase.py` | 4 extraction/generation prompt strings |
| `knox-844` (T7) | `compilation/wicer.py` | 3 WiCER loop prompt strings |
| `knox-lf5` (T8) | `verification.py` | 2 claim verification prompt strings |
| `knox-8gj` (T9) | `output/wiki_index.py` | 1 synthesis prompt string |
| `knox-sga` (T10) | `resolution/llm.py` | 1 default entity extraction prompt |

### Python Antipatterns

1. **Follow the `python-ultimate` skill.** When in doubt about Python coding
   style, structure, or conventions, load and follow the guidelines in the
   `python-ultimate` skill. It takes precedence over any conflicting habit
   or default.

2. **No inline imports — PLC0415 is a hard error.** All imports must live at
   module level (standard library → third-party → local, per PEP 8 grouping).
   Importing inside functions, methods, or class bodies is forbidden. Using
   `# noqa: PLC0415` to suppress the lint error is itself an antipattern — it
   hides a real problem rather than fixing it. Move the import to the top of
   the file. The only accepted exceptions are cases where the import would
   cause a circular dependency or an unconditional heavy side-effect at import
   time; both must be documented with a comment explaining why.

3. **No try/except ImportError guards for core dependencies.** Every package
   listed in `[project.dependencies]` in `pyproject.toml` is guaranteed to be
   installed. Wrapping `import mylib` in `try: ... except ImportError: mylib =
   None` is pointless boilerplate — it obscures the real dependency, adds dead
   branches, and masks actual install failures. Import at module level,
   unconditionally.

4. **All packages are core dependencies unless explicitly marked optional.**
   Unless a package is declared under `[project.optional-dependencies]` in
   `pyproject.toml`, assume it is always available in the venv. Never write
   defensive import logic that treats a required package as optional — if the
   code needs it, it's a dependency.

5. **Install what you need.** When code requires a package that is not yet in
   `pyproject.toml`, add it immediately:
   - `uv add <package>` to install and register the dependency.
   - If `pyproject.toml` was patched manually (e.g. by another agent), run
     `uv sync -U` to bring the venv in sync before importing.
   - Never `import` a package that isn't in the lockfile — the CI and other
     developers will fail.

### Error handling — prefer "let it burn"

The default is to **propagate exceptions**. "Let it burn" means surface real
failures instead of hiding them. The codebase historically over-used
`except: pass` and `except: return None`, converting real errors (bad API
keys, corrupt state, missing deps, disk failures) into silent `None` values.
This forces every caller into `if x is None` chains and produces misleading
messages (e.g. "run compile first" when the real cause is a corrupt cache).

#### When `X | None` is correct vs an antipattern

| `None` means… | Verdict |
|---|---|
| **Absence / lookup miss** (`registry.get(name)`, optional value not set) | ✅ correct |
| **"I tried and failed"** (mute of an exception) | ❌ raise instead |

A function may return `X | None` only when `None` means the thing was
*absent*, never to mean the operation *failed*. `load_cached_wiki` returning
`None` for a **missing** file is correct; returning `None` for a **corrupt**
file is the antipattern.

#### The only valid `try/except` sites

1. **Trust boundary** — translate a low-level error to a user-facing message
   at the CLI/MCP top level. Catch narrowly, report the real cause.
2. **Optional-feature probe** — narrow to the *specific* exception
   (`ImportError`, `FileNotFoundError`), never bare `Exception`.
3. **Best-effort enrichment** — per-item catch that *records* the failure
   (SKIP result, warning log) and continues a batch.
4. **Cache miss** — corrupt/unreadable cache entry treated as a miss is
   correct, BUT narrow to `(json.JSONDecodeError, OSError)` and log at
   `WARNING` so corruption stays visible.

#### Forbidden

- `except Exception: pass` / `except Exception: return None`
- `# noqa: BLE001` or `# noqa: S110` to silence a bare-except lint
- Muting an exception then returning `None` to mean failure
- Catching a corrupt internal-state file (wiki cache, tier state, compile
  state) and silently resetting/recompiling — corruption is a bug; **raise**.

#### Rule of thumb

If removing the `try/except` surfaces a real error to someone who needs to
see it, remove it. If the catch genuinely serves one of the four valid sites,
narrow the exception and make the failure visible (log/record).

### Parameter and default-value antipatterns

These three patterns are **forbidden in new code**. Existing occurrences are
technical debt to be cleaned up incrementally.

#### Antipattern 1 — Hard-coded provider/model names

**Never** hard-code an LLM model identifier as a function default or inline
string literal. Model names are configuration, not code.

    # ❌ FORBIDDEN — model name baked into code
    def _try_litellm_json(prompt, system, model="ollama/llama3.2"):
        ...

    # ❌ FORBIDDEN — same antipattern in __init__, verify_wiki, etc.
    def verify_wiki(wiki, output_dir, *, model="ollama/llama3.2"):
        ...

    # ✅ CORRECT — model comes from config, the only place the default lives
    def _try_litellm_json(prompt, system, model, timeout):
        # model is always provided by the caller, who reads it from LLMConfig
        ...

**Rule:** The string `"ollama/"` (or any other provider prefix) must never
appear outside of:

- `PipelineConfig` / sub-config dataclass field defaults (single source of truth)
- `DEFAULTS` in `_core.py`
- `knox.toml` / `knox.toml.example`
- Test fixtures with an explicit comment `# model name is the system under test`

#### Antipattern 2 — Hard-coded domain values as function defaults

When a parameter's default is a domain-specific value (budget, threshold,
chunk size, iteration count), it **must** come from the config layer, not from
the function signature. The function signature should either accept a grouped
config object or require the caller to pass the value explicitly.

    # ❌ FORBIDDEN — domain defaults scattered across signatures
    def __init__(self, prompt_budget: int = 200_000, extract_budget: int = 6000,
                 concepts_per_batch: int = 8, timeout: float | None = None):

    # ✅ CORRECT — defaults live in one place (the config dataclass)
    @dataclass
    class BudgetConfig:
        prompt: int = 200_000
        extract: int = 6_000
        concepts_per_batch: int = 8

    class TwoPhaseCompiler:
        def __init__(self, config: PipelineConfig) -> None:
            self._budget = config.budget  # single source of truth

#### Antipattern 3 — `| None = None` with hidden default inside the body

When a parameter is declared as `x: SomeType | None = None` and the function
body resolves it to a real default via `x = x or REAL_VALUE` or
`if x is None: x = REAL_VALUE`, the real default is invisible at the call
site and untestable without reading the implementation.

    # ❌ FORBIDDEN — real default (100000) hidden inside the function
    def __init__(self, lint_page_budget: int | None = None) -> None:
        self.lint_page_budget = lint_page_budget or 100000

    # ✅ CORRECT — default is explicit in the type; None is only for
    #              "not provided" at the config layer (DI / testing)
    def __init__(self, budget: BudgetConfig) -> None:
        self.lint_page_budget = budget.lint_page  # default 100_000 in dataclass

If `None` is truly needed (e.g. "use whatever the parent provides"), document
what the effective fallback is in the docstring:

    # Acceptable when None has a well-defined fallback chain
    def __init__(self, state_store: TierStateStore | None = None) -> None:
        """...
        Args:
            state_store: Persistent state handler. When ``None``, defaults to
                ``TierStateStore(output_dir / ".knox" / "compile_state.json")``.
        """

#### Antipattern 4 — Hard cropping to fixed character/token counts

**Never** crop text (LLM output, page content, entity-matching input, prompt
context) to a hard-coded number of characters or tokens. A magic number like
``content[:200]`` silently discards the tail of every input — links, entities,
and facts that live past the cutoff are lost forever, producing false negatives
(orphan pages, missing edges, dropped citations) that look like compiler bugs.

The two failure modes are equally bad:

- **Too small** — real matches beyond the cutoff are silently dropped.
- **Too large** — wastes context but still truncates long inputs.

## ❌ FORBIDDEN — magic character crop

    match_text = (f"{node.title} {[node.content or ''](:200)}").lower()

## ❌ FORBIDDEN — magic token crop (same sin, different unit)

    prompt = text[:4096]

## ✅ CORRECT — scan/process the full content

    match_text = (f"{node.title} {node.content or ''}").lower()

## ✅ CORRECT — when a limit is genuinely needed (model context window

    # output budget), derive it from config, not a literal,
    # and batch/chunk so nothing is silently dropped
    segment_char_size = _tokens_to_chars(cw.max_input - system_tokens - budget)
    segments = _segment_sources(sources, segment_char_size)

**Rule:** If you must bound input size (context window, output token budget),
the limit comes from the model's discovered limits or config — never a literal
in code — and the code must **batch or paginate** so no content is silently
lost. Hard-coded ``[:N]`` / ``[:tokens]`` crops are **forbidden,
non-negotiable**. No exceptions.

### Antipattern 5 — Hard-coded schema/field-name strings

**Never** scatter field names, dict keys, or schema column names across a
module as bare string literals when one site defines the contract (the schema)
and other sites must agree with it (writers, readers, query callers). A typo
in one literal silently desyncs the schema from its readers — a stored field
that never comes back, a query that never matches — and the bug is invisible
until runtime, often far from the cause.

This applies wherever there is a **named, closed set of keys** shared between
a definition and its consumers: vector-store schemas, FTS/SQL column dicts,
metadata keys, serialization tags, API response shapes.

## ❌ FORBIDDEN — schema field names as bare literals

    # schema defines the fields…
    fields=[
        _zvec.FieldSchema("node_id", _zvec.DataType.STRING),
        _zvec.FieldSchema("source_file", _zvec.DataType.STRING),
    ]
    # …but readers repeat the same strings by hand
    source_file = hit.get("source_file")
    node_id = hit.get("node_id")

## ✅ CORRECT — single source of truth as a StrEnum or Final[str] constants

    class ZvecField(StrEnum):
        NODE_ID = "node_id"
        SOURCE_FILE = "source_file"

    _zvec.FieldSchema(ZvecField.NODE_ID, _zvec.DataType.STRING)
    source_file = hit.get(ZvecField.SOURCE_FILE)

**Rule:** When a string literal names a field/column/key that appears in more
than one place (or is part of any persisted schema), it **must** be a
`StrEnum` member (for a closed named set) or a `Final[str]` constant (for a
single value). The string value lives in exactly one definition; every other
site imports the symbol. Bare literals for such names are **forbidden,
non-negotiable**.

**Scope:** This governs our own schema/contract keys only. Keys defined by an
external API or library (e.g. a litellm response ``{"embedding": [...]}``, a
JSON field from an upstream SDK) are that API's contract, not ours to rename —
they are read as literals. When in doubt whether a key is ours, ask: did we
define the schema that produces it? If yes, it is ours and must be a symbol.

### Enforcement — `scripts/check_field_consistency.py`

An AST scanner enforces this antipattern at CI time. Run it with
`mise run check-field-consistency` (or
`uv run python scripts/check_field_consistency.py`). It exits non-zero on any
bare-string literal used as a dict key, subscript, or ``.get()`` argument
that matches a member value of a policed StrEnum (currently ``MetadataField``,
``WikiField``, ``MemoryField``, plus the module-local ``FtsField``/``ZvecField``
scoped to their own modules). The scan is **0 violations** and must stay there.

Notes for working with the checker:

- **Per-enum consumer scoping.** ``MetadataField``/``WikiField`` apply
  repo-wide; ``FtsField``/``ZvecField`` apply only to their defining modules
  (their members — ``"content"``, ``"title"``, ``"id"`` — are too generic to
  police across unrelated dicts like chat-message payloads).
- **Generic values are in a denylist.** ``"source"`` and ``"metadata"`` are
  overloaded across unrelated contracts (edge dicts, parser config, stats) and
  are not policed as bare literals — the enum members remain for explicit use.
- **`# nocheck` per-line opt-out.** When a bare literal legitimately parses an
  external schema (LLM JSON output, checkpoint files, lint-history records),
  append ``# nocheck`` to the line with a short reason. Use sparingly — every
  ``nocheck`` is an unpoliced contract and should be reviewed.
- When you **add a new StrEnum** of contract keys, register it in
  ``ENFORCED_ENUMS`` in the script (with the right consumer scope) so it is
  policed going forward.

### Summary

| Pattern | Forbidden | Correct approach |
|---------|-----------|-----------------|
| `model="ollama/llama3.2"` | ❌ | Config object / explicit param |
| `budget: int = 6000` | ❌ | Config object with grouped defaults |
| `x \| None = None` → `x = x or SECRET` | ❌ | Default in config dataclass, or document fallback in docstring |
| ``content[:200]`` / ``text[:4096]`` | ❌ | Full content, or batch with a config/model-derived limit |
| Bare schema/field-name literals (``hit.get("source_file")``) | ❌ | `StrEnum` / `Final[str]` symbol imported at every site |

#### Jinja2 conventions

- Template files use the `.md.j2` extension (markdown + Jinja2).
- Variables use `{{ snake_case_name }}` syntax.
- Control flow uses `{% for %}` / `{% if %}` blocks, never Python string
  concatenation.
- Each template file **starts with a YAML frontmatter block** delimited by
  `---`. This frontmatter is parsed by `TemplateLoader` and does not appear in
  the rendered output:

---

  variables:
      - name: title
  required: true
      - name: content
  required: true
      - name: context_budget
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

    src/knox/packs/<name>/
    ├── pack.toml           # manifest + prompts (see TOML schema)
    ├── __init__.py         # optional Python registration
    └── templates/          # optional Jinja2 templates
        └── compile.md.j2

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

KNOX produces **OKF v0.1** (Open Knowledge Format) bundles by default.
The spec is at <https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md>

**All output-related code must maintain OKF v0.1 conformance:**

1. **Default format is OKF.** The ``compile.output.format`` config key defaults to
   ``"okf"``. Legacy formats (``"markdown"``, ``"json"``, ``"typst"``) are opt-in.
2. **Every concept file** must have parseable YAML frontmatter with a non-empty
   ``type`` field (OKF \u00a79.2).
3. **Cross-links** use standard markdown absolute bundle-relative paths:
   ``[Title](/concepts/title.md)`` \u2014 never ``[[wikilinks]]`` (OKF \u00a75.1).
4. **Reserved filenames** (``index.md``, ``log.md``) follow OKF structure (\u00a76, \u00a77).
   Root ``index.md`` MAY declare ``okf_version: "0.1"`` in frontmatter (\u00a711).
5. **``log.md``** uses ISO 8601 date headings (``## YYYY-MM-DD``) with bold action
   prefix (\u00a77).
6. **Citations** from provenance are rendered as a ``# Citations`` section with
   numbered ``[N]`` refs (\u00a78).
7. **Consumers MUST tolerate** missing optional fields, unknown ``type`` values,
   unknown frontmatter keys, broken cross-links, and missing ``index.md`` files
   (\u00a79 \u2014 permissive consumption model).

**When adding a new output writer or modifying an existing one**, verify
conformance against the OKF spec. The legacy ``MarkdownWriter`` (Obsidian vault
mode) is exempt from rules 2\u20136 but must remain functional.

### ACP agent — credential-free LLM injection and conversational agents

KNOX implements the **Agent Client Protocol (ACP)** to let editors and AI
agents inject their LLM endpoint without sharing credentials. The agent runs
as ``knox agent`` on stdio.

**Protocol flow:**

1. Client calls ``initialize`` \u2192 KNOX advertises ``providers: {}`` capability
2. Client calls ``providers/list`` \u2192 KNOX returns current LLM endpoint info
3. Client calls ``providers/set`` \u2192 KNOX stores the endpoint and reconfigures litellm
4. Client calls ``session/new`` \u2192 KNOX creates a session
5. Client sends ``/knox <command>`` in a prompt \u2192 KNOX runs the pipeline

**Two prompt paths:**

- **Conversational agent** (default when LLM is configured): prompts go through
  a tool-calling loop driven by TOML-defined agents (``builtin/*.toml`` +
  ``.knox/agents/*.toml``). The LLM reasons over available tools and calls
  them as needed. See ``knox agents list`` for available agents.
- **Subprocess dispatch** (degraded mode, no LLM): ``/knox`` slash commands
  launch the corresponding CLI subcommand. Free-text prompts are rejected.

**Key modules:**

- ``knox_agent.py`` — ACP protocol handler, routes prompts to conversational or subprocess path
- ``definitions.py`` — ``AgentDefinition`` pydantic model loaded from TOML
- ``registry.py`` — ``AgentRegistry`` discovers agents from builtin + vault
- ``tools.py`` — ``ToolDispatcher`` wraps existing MCP server functions as agent-callable tools
- ``conversational.py`` — ``ConversationalAgent`` tool-calling loop
- ``provider_store.py`` — thread-safe LLM credential store for providers/set
- ``slash_commands.py`` — slash command parsing and structured definitions

**Key design decisions:**

- **KNOX controls its own LLM calls.** Unlike MCP sampling (free-text only),
  KNOX uses its full litellm + Pydantic structured output pipeline. The
  ``providers/set`` endpoint configures the LLM; KNOX makes direct HTTP calls.
- **``providers/set`` is an RFD** (not yet in the stable ACP spec). The Python
  SDK v0.11.0 includes the types. When ``providers/set`` is not called, KNOX
  falls back to env vars \u2192 config.toml \u2192 DEFAULTS (unchanged).
- **Slash commands** (``/knox compile``, ``/knox query``, etc.) are parsed from
  the prompt text and dispatched via subprocess. The agent runs on asyncio.
- **Sessions are stateless.** Each prompt turn runs independently. No
  conversation history is accumulated between turns.
- **MCP companion mode is v2.** The existing ``knox mcp`` subcommand is
  unchanged. The ACP agent does not bridge to MCP in v1.

**When modifying the agent module (``src/knox/agent/``):**

- The ``KnoxAgent`` class implements the ``acp.Agent`` protocol. All protocol
  methods are async. Use ``asyncio.run()`` to bridge to synchronous code.
- ``ProviderCredentials`` is thread-safe (``threading.Lock``). It reconfigures
  litellm via env vars (``KNOX_LLM_DEFAULT_MODEL``, ``KNOX_LLM_API_BASE``,
  ``KNOX_LLM_API_KEY``) and restores them on ``disable()``.
- Slash commands are parsed by ``parse_slash_command()`` in
  ``slash_commands.py``. Add new commands there.
- The ``ext_method()`` handler on ``KnoxAgent`` receives unknown ACP methods
  (including ``providers/*``). Add new provider methods there.

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

| Path | Scope |
|------|-------|
| `agent/` | ACP agent — `providers/set` credential injection, TOML-defined conversational agents, tool-calling loop, slash commands |
| `cli/AGENTS.md` | CLI layer — thin shell over core, Typer commands, argument declarations |
| `core/config/AGENTS.md` | Configuration system — resolution, precedence, TOML schema |
| `llm/AGENTS.md` | LLM client — error hierarchy, failover (provider rules delegated to `providers/AGENTS.md`) |
| `preprocessing/parsers/AGENTS.md` | Pluggable ingestion backends — registration, ParserBuildOptions, figure_enrichment semantics |
