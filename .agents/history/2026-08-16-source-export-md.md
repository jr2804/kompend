# Plan — `knox sources export`: regenerate markdown from the parse cache

Created: 2026-08-16
Status: **implemented** (2026-08-16). All phases done; 19 new unit tests pass,
ruff clean on all touched files, ty baseline (203 diagnostics — 0 new from this
work; 2 additions are pre-existing WIP in `compilation/flows/generation.py`),
full unit suite green except pre-existing `test_rel_to_vault.py` collection
error and the WIP's `test_config` gap (fixed by documenting the new keys in
`knox.toml.example`).

## Implementation log (deviations from the plan, on completion)

- **Cache path portability — done at the pipeline layer, not in
  `docling_parser.py`**: the parser has no `base_dir`, so `relativize_image_paths`
  lives in `core/pipeline/ingest.py` and runs in both cache writers
  (`ingest_documents` and `_parse_source` BEFORE the cache write + chunking, so
  parse cache, chunk tree, and asset ids all agree). Idempotent; old caches
  with absolute paths keep working (export resolves both forms).
- **Annotation matching drops the `AssetRecord.source` filter** from the plan:
  ids are content-addressed, so identical tables/figures dedup to one record
  whose summary is content-derived — filtering by source would drop annotations
  for duplicate content in other sources.
- **`knox.chunking` could not be imported** (pre-existing circular import:
  `chunking.summarizer → agents → compilation → assembly → chunking.summarizer`,
  failed under most import orders on clean main). **Fixed** by moving
  `structured_to_text` from `chunking/summarizer.py` to
  `core/models/structured.py` (pure models layer) and repointing the three
  consumers (`chunking/summarizer.py`, `compilation/flows/assembly.py`,
  `compilation/asset_tools.py`, plus the integration test import). `import
  knox.chunking` and `import knox.core.pipeline.runtime` now work standalone.
  `export.py` then switched its duplicated extractor shapes back to the
  canonical `extract_structured`.
- **Equations export without annotations** (as planned): `_asset_kind` returns
  None for FORMULA because the asset store defers equations (Phase 1).
- **`knox.sources export` section written** to `docs/guides/cli.md` +
  `[export]` row in `docs/guides/configuration.md`; `knox.toml.example`
  documents `export.annotations`/`export.summary`.


## Goal

`knox sources export <source-id> <out.md> [--inline|--no-inline]` — a
one-file, clean markdown reconstruction of a source "as ingested", built from
the Tier-1 parse cache (the intermediate JSON), not from the original file.
Attachments are never embedded; they are listed at the end with links.

## What already exists (verified)

| Piece | Location | Note |
|---|---|---|
| Parse cache = serialized `ParsedDocument` (`asdict` + `source_file`), one JSON per source | `vp.cache_dir` = `.knox/cache/`, name `<stem>-<hash8>.json` | `parse_cache_name` / `parse_cache_glob` in `core/pipeline/ingest.py:245/46` |
| Element model | `core/models/elements.py` — `DocumentElement{type, content, page, reading_order, heading_level, metadata, source_file}`; `ElementType` in `core/models/enums.py:135` (text/heading/table/image/formula/list_item/code/metadata) | `content` is `str` (text, headings, formulas, code, image placeholder/path/data-URI) or `list[list[str]]` (tables) |
| Plain-text extraction (drops structure) | `extract_text_from_cached` in `core/pipeline/ingest.py:260` | Not reusable for md; only a model for cache loading |
| Image files from Docling `file` mode | `vp.images_dir` = `.knox/cache/images/`; **absolute** path stored as element content (`docling_parser.py:234`, `str(out_file.resolve())`) | Base64 mode stores data URIs; other parsers store `[Image: caption]` placeholders |
| Attachments manifest + dirs | `AttachmentManifest` in `core/models/attachments.py`; files in `attachments/<source-stem>/` | Manifest is **metadata only** (name, description, ingestible, embed, timestamps) — the attachment files themselves are provided as-is in the dir, never inside the JSONL. `wiki_list_attachments` (mcp/tools/knowledge.py:223) shows the reading pattern |
| Extracted annotations (tables/figures/equations) | `AssetRecord` in `core/models/assets.py`, content-addressed, in `vp.extract_dir` = `.knox/cache/extract/` | Tier-4 knowledge, not Tier-1 ingest output |
| `knox sources` CLI group | `cli/cmds/pipeline/sources.py` (`list`/`tag`/`untag`) | Export joins here |
| Source-id resolution precedent | `cli/cmds/knowledge/capture.py:163` — `cache_dir.glob(parse_cache_glob(stem))` | |

Nothing renders elements → markdown today (verified: `extract_text_from_cached`
concats strings only; `structured_to_text` in `chunking/summarizer.py:205`
works on `StructuredTable` records, not on raw cache tables — not reusable).

## Design (lazy-first)

- **Portability rule: no absolute paths anywhere in the output**, and none
  left in the cache. Docling currently writes absolute image paths into the
  parse cache — fix at the source: `docling_parser.py` stores paths relative
  to `base_dir` (same as `source_file` already is, `ingest.py:207`); export
  resolves them against `base_dir` and copies, so the rendered `.md` only ever
  contains relative references. Output must be movable/archivable as a unit.
- **New module** `src/knox/core/pipeline/export.py` — pure functions, no CLI
  deps: `render_document_markdown(data, *, assets_dir, inline, annotations,
  summary) -> str`, `export_source(cache_file, out_path, *, assets_dir,
  inline, annotations, summary, base_dir) -> Path`,
  `load_cache_entry(source_id, cache_dir) -> dict`. Sibling to `ingest.py`
  (same cache, same pipeline layer).
- **Element → markdown** (one switch over `ElementType`):

  | Element | Output |
  |---|---|
  | text | paragraph (`\n\n`) |
  | heading | `#` × `heading_level` |
  | list_item | `- ` item |
  | code | ``` ```lang ``` ``` fence (lang from metadata if present) |
  | formula | `$$ … $$` |
  | table | GFM pipe table; **first row is always the header** (Docling emits the header row first — no sniffing, no detection) |
  | image | `![caption](target)` — caption from `[Image: x]` placeholder or metadata; target = data URI (inline) / copied file (no-inline mode, relative link) / fallback `[Image]` placeholder text |
  | metadata | skipped (parser internals) |

  Preserve document-level `title`/`author`/`language`/`page_count` as
  frontmatter? **No** — YAGNI; title goes in as `# title` if present, rest
  skipped. Add when requested.
- **Extracted data (annotations + summary) is exported by default, mode
  configurable**: per-item `export.annotations` and `export.summary` in
  `knox.toml`, values `off | inline | asset`, default `inline` ("put in any
  extracted data we have"). CLI overrides: `--annotations` / `--summary` with
  the same three values.

  | Mode | Output |
  |---|---|
  | `off` | skip entirely |
  | `inline` | `## Summary` section right after the title (document-level reads better up front — move to end if preferred); `## Annotated Items` at the very end of the file, one entry per annotated element (asset id, kind, caption, summary, page) |
  | `asset` | JSON file `<out-stem>_assets/extracted.json` — the assets dir is the single on-demand "everything extra" container (images in no-inline mode, extracted.json in asset mode), so the export stays one movable unit; created only when something needs it. Advised over `.knox/assets.json`, which is the internal cross-source store. Shape: `{"source": <source_file>, "summary": ..., "annotations": [{id, kind, caption, summary, page}]}` |

  **Data sources (verified, no new extraction)**:
  - Annotations: `.knox/assets.json` (`AssetStore`, `paths.py:317`), ids
    content-addressed. Recompute ids from parse-cache elements with the
    existing machinery — `chunking/extractors._EXTRACTORS` +
    `asset_registry.asset_id_for` — then filter by `AssetRecord.source ==
    source_file`. Covers tables + figures; equations are not in the store
    (Phase 1 deferral, `asset_registry.py:_kind_of` returns None) — they
    export without annotations until Phase 2 lands.
  - Summary: no persisted per-source summary exists. `_summarize_document`
    (`chunking/contextual.py:139`) is deterministic (headings join, ≤5) and
    cheap — recompute from the parse cache. The LLM extraction summary is
    per-node in a layered content-hash cache, not a document-level value;
    not wired for v1. Add when a consumer asks.
- **Assets**: clean boolean flag `--inline` / `--no-inline`, default
  `--no-inline`. Inline embeds images as data URIs; no-inline copies them to
  `<out-stem>_assets/` next to the output file and references them by relative
  path. Images already on disk are copied; data URIs are decoded to files. One
  flat dir, no per-source nesting. (No `--assets-dir` override — YAGNI, add
  when asked.)
- **Attachments**: always external, never inlined. Read
  `attachments/<source-stem>/` (+ manifest for names/descriptions); list as a
  markdown table at the end of the file: name, description, relative link.
- **Source id**: the `source_file` string printed by `knox sources list`.
  Resolution: glob `parse_cache_glob(stem)` in `cache_dir`; when multiple
  entries match (same-stem files), pick the one whose
  `MetadataField.SOURCE_FILE` equals the id, else error with candidates.
  Also accept a bare stem.
- **CLI**: `export` subcommand on the `sources` group — args: `source_id`,
  `output` (`.md`), `--inline/--no-inline` (default `--no-inline`), `--config`.
- **MCP/API**: out of scope for this plan; CLI covers the use case. Add an MCP
  tool only when a consumer asks.

## Phases

1. **Cache path portability** — `docling_parser.py:234`: store image paths
   relative to `base_dir` instead of `str(out_file.resolve())`. Keeps the
   cache portable and matches how `source_file` is already relativized.
2. **Config keys** — `export.annotations` / `export.summary`
   (`core/config/keys.py` + `_configs.py` + `DEFAULTS` in `generate.py`,
   following the existing pattern); values `off|inline|asset`, default
   `inline`.
3. **Renderer** — `core/pipeline/export.py`: `render_document_markdown` +
  element switch + table/image handling; `export_source` (copies assets,
  writes file, writes attachment table); annotation lookup (extractors +
  `asset_id_for` + `AssetStore`) and summary recompute; inline sections and
  `extracted.json` writer. Resolves image paths against `base_dir`; never
  emits an absolute path.
4. **CLI** — `export` command in `cli/cmds/pipeline/sources.py` +
  `load_cache_entry` id resolution (reuse `_load_all_cache_entries`-style
  loading, or glob single entry).
5. **Tests** — `tests/unit/test_export.py` (new module → new test file):
  - round-trip: build a `ParsedDocument` with one element of every type
    (incl. a table and an image data URI), serialize, render, assert the
    markdown contains each expected form;
  - no-inline mode: image file written to `<stem>_assets/`, relative link
    present, no absolute path anywhere in output;
  - relative image path in cache (post-portability fix) resolves against
    `base_dir` correctly;
  - attachments table rendered with working relative links;
  - same-stem ambiguity: two cache entries, exact `source_file` id wins;
  - annotations inline: element matches an `AssetRecord` (seeded store) →
    entry appears under `## Annotated Items`; no match → no entry;
  - summary inline: `## Summary` present with heading-derived text;
  - `asset` mode for both: `extracted.json` written, correct shape,
    no summary/annotations sections in the `.md`;
  - `off` mode: no sections, no JSON.
6. **Docs + DOX pass** — `docs/` page for the command (after closeout rules
   in root AGENTS.md); update `cli/cmds/pipeline/sources.py` docstring
   sub-command list.

## Verification

```bash
uv run pytest tests/unit/test_export.py -x --timeout=30
ruff check src/knox/core/pipeline/export.py src/knox/cli/cmds/pipeline/sources.py
ty check src/knox
uv run mkdocs build --strict
```

Manual smoke: `knox sources list` → `knox sources export <id> /tmp/out.md` on
an existing vault (PDF with tables/figures) and eyeball the output — then move
`out.md` + `out_assets/` to another directory and confirm links still work
(portability check).
