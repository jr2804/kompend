# Project Initialization

The `knox init` command scaffolds a new wiki project with the standard directory structure, configuration, and templates.

## Usage

```bash
knox init [DIR]
```

If `DIR` is omitted, the current directory is used.

To seed the vault with a chosen config (e.g. a cloud/local LLM profile), pass it
— the file is copied verbatim into `<dir>/.knox/config.toml`:

```bash
knox init ./my-vault --config scripts/config-cloud.toml
```

## What it creates

```text
<dir>/
├── sources/           # Source documents (PDF, DOCX, ...)
├── wiki/              # Compiled wiki output
├── .knox/
│   ├── config.toml    # Project configuration
│   └── cache/         # Runtime cache
├── NOTES.md           # Free-form notes (injected into LLM templates)
├── README.md          # Project readme
└── .gitignore         # Ignores wiki/, .knox/cache/, state files
```

### `.knox/config.toml`

The primary project configuration file. Contains all available settings with documentation comments. See [Configuration](./configuration.md) for the full reference.

### `NOTES.md`

Free-form notes that are injected into LLM templates during compilation. KNOX never modifies this file — it is read-only from KNOX's perspective, preserving your voice. The filename is configurable via `notes.file` in config.

### `README.md`

A starter readme with a brief project description and quick-start instructions.

### `.gitignore`

Standard ignore entries for generated content:

```text
wiki/
.knox/cache/
.knox/state.json
.knox/compile_state.json
```

## Options

### `--bare`

Skip authored templates (`NOTES.md`, `README.md`). Runtime directories and configuration are still created.

```bash
knox init --bare ./my-wiki
```

### `--force`

Overwrite existing project configuration if `.knox/config.toml` or `knox.toml`
already exists. Without this flag, `knox init` prints a warning and exits
gracefully (exit code 0) when a project is detected, rather than scaffolding.

```bash
knox init --force ./existing-project
```

### `--keep-cache`

Only meaningful with `--force`. Preserves `.knox/cache/` while clearing
everything else in the target directory before re-scaffolding. Useful for
fast re-runs where ingest should skip already-parsed documents.

```bash
knox init --force --keep-cache ./existing-project
```

### `--config`

Install a chosen TOML config as the vault's `.knox/config.toml` instead of the
generated default. Use this to seed a vault with a specific LLM/budget/profile
(e.g. a cloud or local demo config). The file is copied verbatim.

```bash
knox init ./my-vault --config scripts/config-cloud.toml
```

### `--git`

Initialize a git repository in the target directory and create an initial commit with all scaffolded files. Without this flag, no git operations are performed.

```bash
knox init --git ./my-wiki
```

## Workflow

```bash
# 1. Initialize
knox init ./my-wiki
cd ./my-wiki

# 2. Add source documents
cp ~/Downloads/paper.pdf sources/
cp ~/Documents/notes.docx sources/

# 3. Compile
knox compile

# 4. Query
knox query "What are the key findings?"
```
