# Attachments

Attachments are supplementary files that belong to a source but are **not** ingested
by default — photographs, binaries, source code, spreadsheets, or anything else
the pipeline shouldn't parse unprompted. Each attachment maintains an explicit
relationship to its parent source.

## How it works

```text
base_dir/
├── sources/              ← documents (auto-ingested)
│   └── report.pdf
├── attachments/          ← attachments (opt-in)
│   └── report/           ← one subdir per source slug
│       ├── photo.jpg
│       └── data.csv
└── .kompend/
    └── kompend.db           ← single store (attachment manifest, with tombstones)
```

**Slug convention**: `attachments/<source-slug>/`, where the slug is the source's
relative path with `/` → `_`. For `report.pdf` → `report/`. For
`papers/thesis.pdf` → `papers_thesis/`.

Extracted figures share this layout: a page's images are copied to the slug of
**the source they came from** (not of the page's first provenance entry, which
may name a venue or product that is not a source at all). One directory per
source holds each figure once; images a page cannot attribute to a source are
not embedded.

## Registering attachments

Use `kompend attachments add` (alias `kompend att add`) to register a file:

```bash
# Register a photo for report.pdf (not ingested)
kompend attachments add report.pdf ./photo.jpg

# Register and mark as ingestible (parsed at compile time)
kompend attachments add report.pdf ./notes.md --ingest

# Override the global embed_attachments setting for this attachment
kompend attachments add report.pdf ./data.csv --ingest --embed
kompend attachments add report.pdf ./private.txt --ingest --no-embed
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--ingest` | Mark as ingestible (parsed during compile). Text/markdown via standard parser; PNG/JPG via VLM caption (if configured). |
| `--embed` / `--no-embed` | Override global `attachments.embed` for this attachment. Without the flag, inherits the global setting. |

## Managing attachments

```bash
# List all attachments (or filter by source)
kompend attachments list
kompend attachments list --source report

# Remove an attachment (tombstone in manifest; file kept on disk)
kompend attachments remove report/photo.jpg

# Read-only check: orphan dirs + manifest rows with no file on disk (no compile needed)
kompend attachments verify

# Register orphaned files on disk (same adoption the compile hook runs; idempotent)
kompend attachments adopt
```

The manifest records one row per path in `.kompend/kompend.db`. Removing an attachment
writes a tombstone row (the file stays on disk); re-adding the same path clears the
tombstone (latest wins).

`kompend attachments verify` reports two findings separately — directories whose slug
matches no source in `sources/`, and manifest rows whose file is missing on disk —
and exits `1` when it finds any, `0` when clean. It is read-only and works before a
compile, unlike the `orphan-attachments` lint rule (which needs a compiled wiki).

## Auto-adoption

At compile start, kompend scans `attachments/<slug>/` for files that have no
manifest entry and registers them automatically:

- `ingestible` depends on `attachments.ingest_adopted` (default `false`):
  text and image extensions become ingestible, binaries are registered
  only. With `false` (the default), adoption only registers the file.
- `embed` inherits the global `attachments.embed` setting.
- Tombstoned paths (explicit `attachments remove`) are never re-adopted.
- Subdirectories whose slug matches no source in `sources/` are skipped —
  the `orphan-attachments` lint rule flags them instead.

This makes recovery from a deleted `.kompend/` lossless for attachments: the
files under `attachments/` are re-registered on the next compile. Run the same
pass on demand with `kompend attachments adopt` (idempotent); check the result
before a compile with `kompend attachments verify`.

## Configuration

```toml
[attachments]
ingest_adopted = true   # adopted text/image files are parsed at compile
embed = false           # include attachment content in the vector index
```

`search.embed_attachments` is gone from the schema — it was renamed to
`attachments.embed`. An existing config still setting the old name is migrated
at resolve time with a deprecation warning (`kompend config check` reports it as
deprecated, not as an unknown key); new configs use `[attachments] embed`.

## Ingestion rules

| Type | Extensions | Ingestible via |
|------|------------|----------------|
| Text | `.md`, `.txt`, `.csv`, `.json`, `.yaml`, `.xml`, `.html`, ... | Standard parser router |
| Image | `.png`, `.jpg` | VLM captioner (requires VLM model configured) |
| Other | `.pdf`, `.docx`, ... | **Rejected** — these are sources, not attachments |
| Binary | `.exe`, `.bin`, ... | Registered only; never ingested |

## Compile behavior

When `--ingest` is set on an attachment:

1. **Parse tier**: text attachments are parsed through the standard router;
   image attachments are captioned via VLM (if `compile.llm.vlm_model` is set,
   otherwise skipped with a warning).
2. **Chunk/compile**: attachment content becomes wiki nodes tagged with
   `source_asset: true` and `parent_source: <slug>` metadata.
3. **Output**: attachments are not duplicated into the compiled wiki — they stay
   in the vault at `attachments/<slug>/<filename>`.

## Embedding (search/RAG)

Set `attachments.embed = true` in `kompend.toml` to include ingestible attachments
in the vector index. Per-attachment `--embed` / `--no-embed` overrides take
precedence.

```toml
[attachments]
embed = true   # default: false
```

When `embed_attachments = false`, attachment nodes are still compiled into the wiki
but excluded from the vector index.

## MCP tools

Five MCP tools provide attachment access to agents:

| Tool | Description |
|------|-------------|
| `wiki_list_attachments(source?)` | List registered attachments (optionally filtered by source). |
| `wiki_read_attachment(source, attachment_name)` | Read text content (up to 100KB) or metadata for binary files. |
| `wiki_attachment_info(source, attachment_name)` | Return manifest entry + file metadata. |
| `attachments_verify(wiki_dir?)` | Read-only check: orphan dirs + manifest rows with no file on disk. |
| `attachments_adopt(wiki_dir?)` | Register orphaned files on disk (compile-hook adoption). |

## Captured and distilled sources

Attachments can be attached to captured or distilled inputs as well:

```bash
kompend attachments add _captured/quick-note.md ./screenshot.png --ingest
```

The source argument is the source slug (relative path with `/` → `_`).
