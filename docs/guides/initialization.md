# Project Initialization

The `kompend init` command scaffolds a new wiki project with the standard directory structure, configuration, and templates.

## Usage

```bash
kompend init [DIR]
```

If `DIR` is omitted, the current directory is used.

To seed the vault with a chosen config (e.g. a cloud/local LLM profile), pass it
— the file is copied verbatim into `<dir>/.kompend/config.toml`:

```bash
kompend init ./my-vault --config scripts/configs/config-cloud.toml
```

## What it creates

```text
<dir>/
├── sources/           # Source documents (PDF, DOCX, ...)
├── wiki/              # Compiled wiki output
├── .kompend/
│   ├── config.toml    # Project configuration
│   └── cache/         # Runtime cache
├── NOTES.md           # Free-form notes (injected into LLM templates)
├── README.md          # Project readme
└── .gitignore         # Ignores wiki/, .kompend/cache/, state files
```

### `.kompend/config.toml`

The primary project configuration file. Contains all available settings with documentation comments; enum-like keys list their remaining valid values in a trailing comment (e.g. `format = "markdown"  # json, typst`). See [Configuration](./configuration.md) for the full reference.

### `NOTES.md`

Free-form notes that are injected into LLM templates during compilation. kompend never modifies this file — it is read-only from kompend's perspective, preserving your voice. The filename is configurable via `notes.file` in config.

### `README.md`

A starter readme with a brief project description and quick-start instructions.

### `.gitignore`

Standard ignore entries for generated content:

```text
wiki/
.kompend/cache/
.kompend/state.json
.kompend/compile_state.json
```

## Options

### `--bare`

Skip authored templates (`NOTES.md`, `README.md`). Runtime directories and configuration are still created.

```bash
kompend init --bare ./my-wiki
```

### `--force`

Overwrite existing project configuration if `.kompend/config.toml` or `kompend.toml`
already exists. Without this flag, `kompend init` prints a warning and exits
gracefully (exit code 0) when a project is detected, rather than scaffolding.

```bash
kompend init --force ./existing-project
```

### `--keep-cache`

Only meaningful with `--force`. Preserves `.kompend/cache/` while clearing
everything else in the target directory before re-scaffolding. Useful for
fast re-runs where ingest should skip already-parsed documents.

```bash
kompend init --force --keep-cache ./existing-project
```

### `--config`

Install a chosen TOML config as the vault's `.kompend/config.toml` instead of the
generated default. Use this to seed a vault with a specific LLM/budget/profile
(e.g. a cloud or local demo config). The file is copied verbatim.

```bash
kompend init ./my-vault --config scripts/configs/config-cloud.toml
```

### `--wiki-dir` / `-w`

Scaffold into the given directory as an alternative to the `DIR` argument —
`kompend init -w ./my-wiki` is equivalent to `kompend init ./my-wiki`. This
mirrors the global `--wiki-dir` option (which must precede the subcommand).
Passing both a `DIR` and `--wiki-dir` with different paths is an error.

```bash
kompend init --wiki-dir ./my-wiki
```

### `--git`

Initialize a git repository in the target directory and create an initial commit with all scaffolded files. Without this flag, no git operations are performed.

```bash
kompend init --git ./my-wiki
```

## Workflow

```bash
# 1. Initialize
kompend init ./my-wiki
cd ./my-wiki

# 2. Add source documents
cp ~/Downloads/paper.pdf sources/
cp ~/Documents/notes.docx sources/

# 3. Compile
kompend compile

# 4. Query
kompend query "What are the key findings?"
```
