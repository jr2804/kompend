# CLI Reference

KNOX provides a Typer-based CLI with the following commands:

| Command | Description |
|---------|-------------|
| `knox parse` | Parse documents and show their structure |
| `knox compile` | Compile documents into a knowledge wiki |
| `knox doctor` | Validate configuration and connectivity |
| `knox status` | Show per-wiki statistics |
| `knox list-parsers` | List all registered parsers |

## Global Options

| Flag | Description |
|------|-------------|
| `--version`, `-v` | Show version and exit |
| `--help` | Show help message and exit |

## `knox parse`

Parse documents and display their structural elements.

```
knox parse [OPTIONS] FILES...
```

**Arguments:**
| Name | Description |
|------|-------------|
| `FILES...` | Files, glob patterns, or directories to parse (required) |

**Options:**
| Flag | Description |
|------|-------------|
| `--recursive/--no-recursive`, `-r/-R` | Search source directories recursively |
| `--elements`, `-e` | Print extracted elements |

## `knox compile`

Compile documents into a knowledge wiki.

```
knox compile [OPTIONS] FILES...
```

**Arguments:**
| Name | Description |
|------|-------------|
| `FILES...` | Files, glob patterns, or directories to compile (required) |

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--sources-dir`, `-s` | `sources` | Input sources directory |
| `--output-dir`, `-o` | `wiki` | Output directory for compiled wiki |
| `--url` | — | URL to download before processing (repeatable) |
| `--recursive/--no-recursive` | no-recursive | Search directories recursively |
| `--format`, `-f` | `markdown` | Output format: markdown, json, typst |
| `--chunk-size` | `512` | Chunk size in tokens |
| `--image-mode` | `file` | Image handling: file, base64 |
| `--index` | — | Index compiled wiki for semantic search |
| `--embed-model` | — | Embedding model for vector indexing |
| `--ignore` | — | Ignore pattern (glob-style, repeatable) |
| `--progress/--no-progress` | auto | Show rich progress bars (auto-detected for TTY) |

### Ignore patterns

Files matching ignore patterns are excluded from compilation. Patterns use
`.gitignore`-style glob syntax:

```bash
knox compile *.pdf --ignore "*.bak" --ignore "drafts/**"
```

Ignore patterns can also be configured in `.knox/config.yaml`:
```yaml
ignore:
  - "*.tmp"
  - "drafts/*"
  - "**/private/*.pdf"
```

By default, `.gitignore` in the source directory is respected. Set
`ignore_gitignore: false` in config to disable.

### Progress bars

When stdout is a TTY, `knox compile` shows rich progress bars:

- **Across-file**: `Parsing sources [45/50] ████████████████░░░░ 90% 0:02:34`
- **Within-file**: Pipeline steps per file (Download, Parse, Chunk, Resolve,
  Compile, Write)

Use `--no-progress` to disable, or `--progress` to force enable.

## `knox doctor`

Validate configuration and connectivity.

```
knox doctor [OPTIONS]
```

**Options:**
| Flag | Description |
|------|-------------|
| `--verbose` | Show detailed output (including LLM connectivity check) |
| `--json` | Output in JSON format for machine readability |

Checks performed:
1. **Source directory** — exists and is a directory
2. **Output directory** — exists and is writable
3. **Parsers** — all registered parser dependencies load correctly
4. **Git repository** — check if inside a git repo
5. **Disk space** — warn if < 500 MB free
6. **Environment variables** — KNOX_LLM_MODEL, KNOX_LLM_PROVIDER, OPENAI_API_KEY
7. **LLM connectivity** — quick ping to configured LLM (only with `--verbose`)

Exit code: `0` if all checks pass, `1` if any check fails.

## `knox status`

Show per-wiki statistics.

```
knox status [OPTIONS]
```

**Options:**
| Flag | Description |
|------|-------------|
| `--json` | Output in JSON format |

Displays:
- Wiki name and config file path
- Source count and total size
- Article count and total word count
- Last compile timestamp
- Health score (if eval is set up)

## `knox list-parsers`

List all registered parsers (built-in, external, and HTTP).

```
knox list-parsers [OPTIONS]
```

**Options:**
| Flag | Description |
|------|-------------|
| `--json` | Output in JSON format |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `KNOX_SOURCES_DIR` | Default sources directory |
| `KNOX_OUTPUT_DIR` | Default output directory |
| `KNOX_RECURSIVE` | Default recursive mode (true/false) |
| `KNOX_LLM_MODEL` | Default LLM model |
| `KNOX_LLM_PROVIDER` | Default LLM provider |
