# Configuration

KNOX uses a layered configuration system. Values are resolved from multiple sources, with CLI flags taking the highest priority and built-in defaults the lowest.

## 5-Layer Precedence

```
1. CLI flags (--wiki-dir, --format, ...)         ← highest
2. Environment variables (KNOX_*)
3. Project config (.knox/config.toml + knox.toml)
4. Global config (~/.config/knox/config.toml)
5. Built-in defaults                              ← lowest
```

### Layer 1: CLI flags

Most options can be passed directly on the command line. See `knox <command> --help` for available flags.

### Layer 2: Environment variables

All environment variables use the `KNOX_` prefix and map to dotted config keys:

| Variable | Config Key |
|----------|------------|
| `KNOX_WIKI_DIR` | `paths.wiki_dir` |
| `KNOX_SOURCES_DIR` | `paths.sources_dir` |
| `KNOX_OUTPUT_DIR` | `paths.output_dir` |
| `KNOX_NOTES_FILE` | `notes.file` |
| `KNOX_LLM_MODEL` | `compile.llm.default_model` |
| `KNOX_VLM_MODEL` | `compile.llm.vlm_model` |

### Layer 3: Project config (hierarchical)

KNOX searches for project config files by walking up from the current directory to the filesystem root. Two filenames are recognized:

- **`.knox/config.toml`** — the primary project config (created by `knox init`)
- **`knox.toml`** — legacy fallback, supported for backward compatibility

If multiple files exist in the hierarchy, they are merged with **closest-to-CWD wins** semantics. Deep merge at the section level; individual keys are overwritten (lists replaced, not concatenated).

Example hierarchy:

```
/home/user/projects/monorepo/
├── .knox/config.toml          ← parent config (shared defaults)
└── packages/
    └── my-wiki/
        └── .knox/config.toml  ← child config (overrides parent)
```

Running `knox compile` from `my-wiki/` resolves both files; the child's keys win.

### Layer 4: Global config

A single global config file at `~/.config/knox/config.toml` (or `%APPDATA%\knox\config.toml` on Windows). Used for machine-wide defaults like LLM model selection.

### Layer 5: Defaults

Built-in defaults ship with the package. Inspect them via `knox config list`.

## Managing Configuration

### `knox config get`

```bash
knox config get compile.format
knox config get paths.wiki_dir
```

Prints the resolved value, respecting all 5 layers.

### `knox config set`

```bash
# Write to project config (.knox/config.toml)
knox config set compile.format json
knox config set compile.chunk_size 1024

# Write to global config
knox config set --global compile.llm.default_model gpt-4o
```

`set` validates keys against the schema, coerces types (int, float, bool, string, list), writes atomically, and creates a `.bak` backup before mutating.

### `knox config list`

```bash
knox config list
```

Prints all resolved keys with provenance — which file or source supplied each value.

### `knox config path`

```bash
knox config path           # project config path
knox config path --global  # global config path
```

## Configuration Reference

The main configuration sections:

| Section | Purpose |
|---------|---------|
| `[paths]` | Wiki directory, source/output/cache/state paths |
| `[compile]` | Compile settings: format, chunk size, image mode, workers |
| `[compile.budget]` | Token budget settings |
| `[compile.llm]` | LLM model, provider, fallback chain, role models |
| `[compile.embedding]` | Embedding model for vector index |
| `[compile.memory]` | Memory lifecycle (confidence decay, tiers) |
| `[notes]` | User notes file configuration |
| `[hooks]` | Event hook callbacks |

The full annotated reference is available in the config template created by `knox init` at `.knox/config.toml`.

## Migration from `--sources-dir` / `--output-dir`

In earlier versions, path flags were per-command (`knox compile --sources-dir X --output-dir Y`). These have been replaced by the global `--wiki-dir` flag.

**Before:**

```bash
knox compile --sources-dir ./my_docs --output-dir ./output
```

**After:**

```bash
# Set the project root; sources/ and wiki/ are derived from it
knox --wiki-dir ./my-project compile

# Or configure via .knox/config.toml
[paths]
sources_dir = "my_docs"
output_dir = "output"
```

Individual paths can still be overridden via config or environment variables for advanced use cases.
