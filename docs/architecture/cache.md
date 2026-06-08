# Cache and State Directory

KNOX stores derived data, pipeline state, and indexes inside a `.knox/`
directory within the output directory. This document describes what goes
where, why, and the design rules for new caches.

## Directory layout

Default output directory is `wiki/`. All state lives under `wiki/.knox/`:

```
wiki/
├── .knox/
│   ├── last_wiki.json          # Serialized CompiledWiki (N1.5)
│   ├── compile_state.json      # Tier state tracking (Q20)
│   ├── cache/                  # Parse result cache
│   │   ├── a1b2c3d4.json       # Per-source parsed document
│   │   └── ...
│   ├── vectors/                # Zvec vector index (RAG)
│   ├── review/                 # Review queue entries (N8)
│   │   ├── node_0.json
│   │   └── ...
│   ├── provenance/             # Per-node citation data (N4)
│   │   ├── intro.json
│   │   └── ...
│   └── private.json            # Private page index (Q21)
├── index.md                    # Auto-generated wiki index
├── log.md                      # Compilation log
└── ...                         # Compiled wiki pages
```

## What is stored and why

| Path | Contents | Typical size | Scope | Created by |
|------|----------|-------------|-------|-----------|
| `last_wiki.json` | Full `CompiledWiki` dataclass as JSON | 10 KB – 10 MB | Project-local | Pipeline after compile |
| `compile_state.json` | Per-file tier progress (parsed, chunked, etc.) | < 50 KB | Project-local | `TierStateStore` |
| `cache/*.json` | Cached `ParsedDocument` per source file (hash-keyed) | 100 KB – 100 MB | Project-local | Pipeline parse tier |
| `vectors/` | Zvec embedded vector database for semantic search | 1 – 100 MB | Project-local | `ZvecWikiStore` |
| `review/*.json` | Wiki nodes pending human review | < 100 KB | Project-local | `ReviewQueue` |
| `provenance/*.json` | Per-node citation/cross-ref data | < 500 KB | Project-local | `annotate_wiki_with_provenance` |
| `private.json` | Index of pages marked private | < 10 KB | Project-local | `mark_wiki_private` |

## Project-local vs. global: design rule

**All KNOX caches are project-local and should stay that way.**

A cache belongs in `.knox/` (inside the output directory) when:
- It contains **derived data** specific to a particular source corpus
- It is **not reusable** across different projects
- Rebuilding it requires the original source files

Global caches (e.g. `~/.cache/knox/`) would only make sense for:
- **Downloaded ML models** — but HuggingFace, sentence-transformers, and
  Docling already manage their own caches under `~/.cache/huggingface/`
  and `~/.cache/docling/`
- **Parser binaries** — pandoc, external tools — managed by the OS package
  manager or pipx

KNOX should **not** re-invent model or tool caching. The `.knox/` directory
is for KNOX-derived data only.

## Cache lifecycle

### Creation

`.knox/` is created on demand during `knox compile`. It is **not** versioned
in git — the directory should be added to `.gitignore` (it is by default in
the KNOX project template).

### Invalidation

| Cache | When it becomes stale | What to do |
|-------|----------------------|-----------|
| `cache/*.json` | Source file modified or `--no-cache` flag | Delete `.knox/cache/` or re-run compile |
| `compile_state.json` | Any source file changes | `--force` flag re-processes everything |
| `last_wiki.json` | After every successful compile | Overwritten automatically |
| `vectors/` | After compile changes node content | Re-run `--index` |
| `review/` | After review actions | Entries removed as they are approved/rejected |

### Cleanup

```bash
# Nuke all cached state
rm -rf wiki/.knox/

# Re-parse everything from scratch
knox compile ./sources --no-cache --force
```

## Adding a new cache

When adding a new subsystem that needs persistent state:

1. **Place it under `<output_dir>/.knox/<subsystem>/`**
2. **Document it in the table above** (this file)
3. **Make it resilient to missing data** — if the cache directory is
   deleted, KNOX should rebuild from source files, not crash
4. **Never store source files or user data** — only derived/compiled data
5. **Use the PipelineConfig.cache_dir path** when available, or construct
   the path as `output_dir / ".knox" / "<subsystem>"`

## External caches (not managed by KNOX)

These are managed by upstream libraries and listed here for reference:

| Path | What | Managed by |
|------|------|-----------|
| `~/.cache/huggingface/` | sentence-transformers models (≈1 GB each) | HuggingFace Hub |
| `~/.cache/docling/` | Docling model files (≈500 MB) | Docling |
| `~/.cache/litellm/` | LiteLLM cache (if enabled) | LiteLLM |
| `~/.cache/pip/` | Python packages | pip |

KNOX does not control, configure, or clean these. Users manage them through
their respective tools (e.g. `huggingface-cli delete-model`, `pip cache purge`).

## Config file location

Config files are **separate** from the cache directory:

| File | Purpose | Tracked in git |
|------|---------|---------------|
| `knox.toml` | Project config | No (gitignored) |
| `knox.toml.example` | Reference config | Yes |
| `~/.config/knox/config.toml` | User global config | No |

See `knox.core.config` AGENTS.md for the full config precedence system.
