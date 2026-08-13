# Assets (Attachments)

Assets are supplementary files that belong to a source but are **not** ingested
by default — photographs, binaries, source code, spreadsheets, or anything else
the pipeline shouldn't parse unprompted. Each asset maintains an explicit
relationship to its parent source.

## How it works

```text
base_dir/
├── sources/              ← documents (auto-ingested)
│   └── report.pdf
├── assets/               ← attachments (opt-in)
│   └── report/           ← one subdir per source slug
│       ├── photo.jpg
│       └── data.csv
└── .knox/
    └── assets.jsonl      ← manifest (append-only)
```

**Slug convention**: `assets/<source-slug>/`, where the slug is the source's
relative path with `/` → `_`. For `report.pdf` → `report/`. For
`papers/thesis.pdf` → `papers_thesis/`.

## Registering assets

Use `knox assets add` to register a file as an attachment:

```bash
# Register a photo for report.pdf (not ingested)
knox assets add report.pdf ./photo.jpg

# Register and mark as ingestible (parsed at compile time)
knox assets add report.pdf ./notes.md --ingest

# Override the global embed_assets setting for this asset
knox assets add report.pdf ./data.csv --ingest --embed
knox assets add report.pdf ./private.txt --ingest --no-embed
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--ingest` | Mark as ingestible (parsed during compile). Text/markdown via standard parser; PNG/JPG via VLM caption (if configured). |
| `--embed` / `--no-embed` | Override global `search.embed_assets` for this asset. Without the flag, inherits the global setting. |

## Managing assets

```bash
# List all assets (or filter by source)
knox assets list
knox assets list --source report

# Remove an asset (tombstone in manifest; file deleted from assets/)
knox assets remove report/photo.jpg
```

The manifest is append-only JSONL. Removing an asset appends a tombstone entry;
re-adding the same path creates a new active entry (latest wins).

## Ingestion rules

| Type | Extensions | Ingestible via |
|------|------------|----------------|
| Text | `.md`, `.txt`, `.csv`, `.json`, `.yaml`, `.xml`, `.html`, ... | Standard parser router |
| Image | `.png`, `.jpg` | VLM captioner (requires VLM model configured) |
| Other | `.pdf`, `.docx`, ... | **Rejected** — these are sources, not assets |
| Binary | `.exe`, `.bin`, ... | Registered only; never ingested |

## Compile behavior

When `--ingest` is set on an attachment:

1. **Parse tier**: text attachments are parsed through the standard router;
   image attachments are captioned via VLM (if `compile.llm.vlm_model` is set,
   otherwise skipped with a warning).
2. **Chunk/compile**: attachment content becomes wiki nodes tagged with
   `source_asset: true` and `parent_source: <slug>` metadata.
3. **Output**: all registered assets are copied to `wiki/assets/` so generated
   content can link to them with relative paths (`assets/<slug>/<filename>`).

## Embedding (search/RAG)

Set `search.embed_assets = true` in `knox.toml` to include ingestible
attachments in the vector index. Per-asset `--embed` / `--no-embed` overrides
take precedence.

```toml
[search]
embed_assets = true   # default: false
```

When `embed_assets = false`, attachment nodes are still compiled into the wiki
but excluded from the vector index.

## MCP tools

Three MCP tools provide asset access to agents:

| Tool | Description |
|------|-------------|
| `wiki_list_assets(source?)` | List registered attachments (optionally filtered by source). |
| `wiki_read_asset(source, asset_name)` | Read text content (up to 100KB) or metadata for binary files. |
| `wiki_asset_info(source, asset_name)` | Return manifest entry + file metadata. |

## Captured and distilled sources

Assets can be attached to captured or distilled inputs as well:

```bash
knox assets add _captured/quick-note.md ./screenshot.png --ingest
```

The source argument is the source slug (relative path with `/` → `_`).
