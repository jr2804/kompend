# Configuration

kompend uses a layered configuration system. Values are resolved from multiple sources, with CLI flags taking the highest priority and built-in defaults the lowest.

## 5-Layer Precedence

```text
1. CLI flags (--wiki-dir, --set, model swaps, ...)   ← highest
2. Environment variables (KOMP_*)
3. Project config (.kompend/config.toml + kompend.toml) — discovered from --wiki-dir when set
4. Global config (~/.config/kompend/config.toml)
5. Built-in defaults                              ← lowest
```

### Layer 1: CLI flags

Most options can be passed directly on the command line. See `kompend <command> --help` for available flags.

### Layer 2: Environment variables

All environment variables use the `KOMP_` prefix and map to dotted config keys:

| Variable | Config Key |
|----------|------------|
| `KOMP_WIKI_DIR` | `paths.wiki_dir` |
| `KOMP_SOURCES_DIR` | `paths.sources_dir` |
| `KOMP_OUTPUT_DIR` | `paths.output_dir` |
| `KOMP_NOTES_FILE` | `notes.file` |
| `KOMP_LLM_DEFAULT_MODEL` | `compile.llm.default_model` |
| `KOMP_VLM_MODEL` | `compile.llm.vlm_model` |
| `KOMP_SEARCH_EXPANSION_ENABLED` | `search.expansion.enabled` |
| `KOMP_SEARCH_EXPANSION_VARIANTS` | `search.expansion.variants` |
| `KOMP_SEARCH_CRAG_ENABLED` | `search.crag.enabled` |
| `KOMP_SEARCH_CRAG_THRESHOLD` | `search.crag.threshold` |
| `KOMP_SEARCH_RERANKER_ENABLED` | `search.reranker.enabled` |
| `KOMP_SEARCH_RERANKER_MODEL` | `search.reranker.model` |
| `KOMP_SEARCH_RERANKER_TOP_K` | `search.reranker.top_k` |
| `KOMP_SEARCH_HYDE_ENABLED` | `search.hyde.enabled` |
| `KOMP_SEARCH_CONTEXTUAL_ENABLED` | `search.contextual.enabled` |
| `KOMP_SEARCH_CONTEXTUAL_MODEL` | `search.contextual.model` |

### Layer 3: Project config (hierarchical)

kompend searches for project config files by walking up from the current directory to the filesystem root. Two filenames are recognized:

- **`.kompend/config.toml`** — the primary project config (created by `kompend init`)
- **`kompend.toml`** — legacy fallback, supported for backward compatibility

Each committed config can be overlaid by a **local, gitignored twin** —
`.kompend/config.local.toml` next to `.kompend/config.toml`, `kompend.local.toml`
next to `kompend.toml` (mise-style, created by appending `.local` to the
filename). Local values win over the committed config in the same
directory. Use it for machine-specific secrets like provider API keys;
`kompend init` ignores these files in the generated `.gitignore`.

If multiple files exist in the hierarchy, they are merged with **closest-to-CWD wins** semantics. Deep merge at the section level; individual keys are overwritten (lists replaced, not concatenated).

Example hierarchy:

```text
/home/user/projects/monorepo/
├── .kompend/config.toml          ← parent config (shared defaults)
└── packages/
    └── my-wiki/
        └── .kompend/config.toml  ← child config (overrides parent)
```

Running `kompend compile` from `my-wiki/` resolves both files; the child's keys win.

### Layer 4: Global config

A single global config file at `~/.config/kompend/config.toml` (or `%APPDATA%\kompend\config.toml` on Windows). Used for machine-wide defaults like LLM model selection.

### Layer 5: Defaults

Built-in defaults ship with the package. Inspect them via `kompend config list`.

## Managing Configuration

### `kompend config get`

```bash
kompend config get compile.output.format
kompend config get paths.wiki_dir
```

Prints the resolved value, respecting all 5 layers.

### `kompend config set`

```bash
# Write to project config (.kompend/config.toml)
kompend config set compile.output.format json
kompend config set compile.chunk_size 1024

# Write to global config
kompend config set --global compile.llm.default_model gpt-4o
```

`set` validates keys against the schema, coerces types (int, float, bool, string, list), writes atomically, and creates a `.bak` backup before mutating.

### `kompend config list`

```bash
kompend config list
```

Prints all resolved keys with provenance — which file or source supplied each value.
Provider `api_key` values are redacted (`***`) unless they are `${ENV_VAR}` references.

### `kompend config path`

```bash
kompend config path           # project config path
kompend config path --global  # global config path
```

### `kompend config check`

Diff a TOML config file against the built-in schema defaults. Useful when
upgrading kompend or migrating between versions — the report lists every key
that differs from the shipped default plus any unknown keys or
validation errors.

```bash
# Discover the project config automatically.
kompend config check

# Use a specific vault's config.
kompend -w wiki config check

# Explicit file — exits 0 clean, 1 if any modified, 2 if any errors.
kompend config check ./kompend.toml

# Multiple files — merged (later wins); reports one combined diff.
kompend config check ./base.toml ./local.toml

# Independent per-file reports.
kompend config check ./base.toml ./local.toml --no-merge

# Machine-readable JSON.
kompend config check --json
```

Status legend:

| Status | Meaning |
|--------|---------|
| `ok`       | Value matches the schema default. |
| `modified` | User value differs from the default (expected for any customised key). |
| `error`    | Unknown key, or wrong type/value (pydantic validation failed). |

The exit code is `0` (clean), `1` (only modified), or `2` (errors), so
this command plugs straight into CI to catch typos in checked-in config.

### Duplicate keys

A key assigned twice in the same table (a common hand-merge artifact)
no longer discards the whole file. Kompend keeps the first occurrence,
drops the later duplicates, and logs a warning naming each dropped key
and its line number. Adjacent re-declarations of the same `[table]`
header are merged the same way; any other TOML error still fails the
file load.

### Removed keys

A config key either has an effect or does not exist. Keys without a
reader were removed from the schema; an old vault that still carries one
gets a warning and the key is ignored — the file keeps loading.

| Removed key | Why |
|-------------|-----|
| `compile.mode` | The tier flags (`start_at_tier` / `stop_at_tier`) cover the same ground and are the only ones the pipeline reads. |
| `compile.llm.fallback_max_retries` | Unused; the fallback chain is tried in order. |
| `compile.llm.role_models` | Superseded by `compile.llm.default_model`. |
| `eval.threshold`, `eval.history_size` | The evaluation loop no longer exists. |
| `display.error_chars`, `display.json_chars` | Never read — the renderers use their own formatting. |
| `display.query_expansions` | No expander behind the key. |
| `display.wiki_topics` | Fed an index section that was dropped. |
| `ingest.parser`, `ingest.parser_fallback` | Parsecraft is the only backend and routes internally. |
| `compile.pdf.*` | The kompend-owned docling/liteparse backends were replaced by parsecraft. |

`kompend config check` reports them as deprecations.

## Configuration Reference

The main configuration sections:

| Section | Purpose |
|---------|---------|
| `[paths]` | Wiki directory, source/output/cache/state paths; `recursive` controls whether `sources/` subdirectories are scanned (`--recursive/--no-recursive`) |
| `[compile]` | Compile settings: format, chunk size, image mode, workers |
| `[compile.budget]` | Token budget settings — budgets shape the work, they never exclude it (see [Model Context Adaptation](model-context.md) for full reference) |
| `[compile.llm]` | LLM model, provider, fallback chain, role models |
| `[compile.embedding]` | Embedding model for the vector index (`<provider>:<model>`, e.g. `ollama:granite-embedding:278m`; empty disables) |
| `[compile.memory]` | Memory lifecycle (confidence decay, tiers) |
| `[display]` | Output limits (excerpts, snippets, description length, CLI previews) |
| `[ingest]` | Ingestion settings; `parsecraft_backend_options` overrides the parsecraft backend options |
| `[providers.<id>]` | User-defined OpenAI-compatible endpoints (self-hosted llama.cpp, vLLM, TGI). See [LLM Configuration](llm-config.md#user-defined-providers) |
| `[export]` | `kompend sources export` placement of extracted data (`annotations` / `summary`: `off` \| `inline` \| `asset`) |
| `[notes]` | User notes file configuration |
| `[hooks]` | Event hook callbacks |

The full annotated reference is available in the config template created by `kompend init` at `.kompend/config.toml`. Enum-like keys carry a trailing comment listing the remaining valid values (e.g. `format = "markdown"  # json, typst`).

## Migration from `--sources-dir` / `--output-dir`

In earlier versions, path flags were per-command (`kompend compile --sources-dir X --output-dir Y`). These have been replaced by the global `--wiki-dir` flag.

**Before:**

```bash
kompend compile --sources-dir ./my_docs --output-dir ./output
```

**After:**

```bash
# Set the project root; sources/ and wiki/ are derived from it
kompend --wiki-dir ./my-project compile

# Or configure via .kompend/config.toml
[paths]
sources_dir = "my_docs"
output_dir = "output"
```

Individual paths can still be overridden via config or environment variables for advanced use cases.
