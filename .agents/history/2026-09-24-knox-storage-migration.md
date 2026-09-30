# `.knox/` Storage Inventory + Single-Store Migration Spec

TASK-4 deliverable. All producer/consumer citations verified against code
(`src/knox/**`, file:line at time of writing). On-disk sizes cross-checked
against the live vault `C:\Projects\Tech\p863-rev-eng\wiki\.knox\`.

**Status: implemented (archived).** The design below was accepted in full:
engine = single SQLite store (`A1`), migration = clean cut with no migration code
(`B3`). It shipped as `61147c5f` (foundation: `core/store/` package, schema,
versioning, `.knox/knox.db` location) and `f6aee4c0` (consolidation: every
non-vector artifact moved into the store, 12 dead readers repointed). Two
configuration TOMLs (`.knox/config.toml`, `.knox/rumdl.toml`) stay
human-editable TOML; the zvec `vectors/` directory stays the optional
regenerable embedding branch. Sections 1–2 are retained as the artifact
inventory and table mapping that the implementation was built from; the
engine-choice and migration-strategy options in §3–§4 are historical.

---

## 1. Complete artifact inventory

Every path below is defined in `src/knox/core/paths.py`. Layout reference:
`paths.py:207–227`. `VaultPaths.ensure_dirs()` (`paths.py:437–455`) pre-creates:
`sources/`, `wiki/`, `.knox/`, `cache/`, `dedup/`, `cache/images/`,
`cache/extract/`, `provenance/`, `review/`, `vectors/`.

### 1.1 Config files — must stay human-editable TOML

| Artifact | Constant | Producer | Consumer | Notes |
|---|---|---|---|---|
| `.knox/config.toml` | `CONFIG_FILE` / `KNOX_CONFIG_FILE` (paths.py:52–53) | user-authored; scaffolded by `knox init` (`cli/cmds/init.py:40`) | config resolution (`core/config/_core.py`) | also a vault marker (`VAULT_MARKERS`, paths.py:57) |
| `.knox/rumdl.toml` | `RUMDL_CONFIG_FILE` (paths.py:58) | materialized from shipped default on first use; `# knox-managed: <hash>` marker preserves user edits below it (`core/config/rumdl.py:53–75`) | `resolve_rumdl_config` → markdown normalization | knox-managed but carries a user-editable section |

### 1.2 Loose JSON files (directly under `.knox/`)

| # | Artifact | Constant | JSON shape (top level) | Producer (write) | Consumer (read) | Class | Regeneration |
|---|---|---|---|---|---|---|---|
| 1 | `compile_state.json` | `COMPILE_STATE_FILE` (paths.py:104) | `{"entries": {stem: {stem, source_file, content_hash, current_tier:int, last_compile, search_hits:int, last_hit, ingested_by}}}` (`core/pipeline/state.py:31–98`) | `TierStateStore.save` via `core/pipeline/ingest.py:154`; rewritten by organize renames `organize/moves.py:157` | `cli/cmds/pipeline/status.py:74`, `lint/structural.py:646`, `core/pipeline/stages/_index.py:39`, organize moves | **derived-but-costly** | lost ⇒ every source recompiles from tier 0 (full LLM re-extraction) |
| 2 | `last_wiki.json` | `WIKI_CACHE_FILE` (paths.py:67) | `{nodes:[WikiNode…], edges, toc, quality_score, iteration, metadata}` (`core/cache/wiki.py:41–51`) | `write_wiki_cache` (`core/cache/wiki.py:20`), called post-compile | `cli/cmds/knowledge/capture.py:321`, `core/models/wiki.py:140`, `mcp/tools/pipeline.py:297`, `organize/moves.py:234`, `web/_view_helpers.py:43`, lint/log/query paths | **derived** | `knox compile` |
| 3 | `compiled_wiki.json` | `COMPILED_WIKI_FILE` (paths.py:68) | same shape as #2 (`core/pipeline/stages/_compile.py:49–72`) | `save_compiled_wiki` (`stages/_compile.py:35,49`), at compile-tier end | `load_compiled_wiki` (`stages/_compile.py:74`), resumed by `stages/_pipeline.py:200` | **derived** | `knox compile` — **redundant near-duplicate of #2; reconcile/dedup during migration** |
| 4 | `memory.json` | `MEMORY_STORE_FILE` (paths.py:89) | `{"states": [{node_id, title, confidence:float, tier, birth_at, review_at, review_count:int, superseded_by}]}` (`memory/lifecycle.py:56–101`) | `update_memory_for_compile` (`memory/lifecycle.py:296–302`), post-compile (`stages/_pipeline.py:39`) | lint memory-lifecycle checks (`lint/structural.py:480–530`); next compile update pass | **semi-durable** | rebuildable by compile, but loss resets tier promotion / decay clocks (user-visible page prominence) |
| 5 | `attachments.jsonl` | `ATTACHMENTS_MANIFEST` (paths.py:117) | JSONL; record `{path, source, ingestible:bool, embed, registered_at, deleted:bool}`; **append-only, latest-per-path wins, `deleted:true` = tombstone** (`core/models/attachments.py:173–244,288–310`) | `AttachmentManifest.append` — `cli/cmds/knowledge/attachments.py:96,127,198,243`, `core/pipeline/attachments.py:46`, `stages/_parse.py:410`, `mcp/tools/attachments.py:60,90,158`, `mcp/tools/knowledge.py:286,372` | `read_all`/`read_for_source`/`deleted_paths` — `core/pipeline/export.py:435`, `lint/structural.py:713`, adoption gate `core/models/attachments.py:330–366` | **durable / irrecoverable** | **NONE** — tombstones record explicit user removals; loss lets `attach_orphan_attachments()` re-adopt removed files |
| 6 | `lint_history.jsonl` | `LINT_HISTORY_FILE` (paths.py:108) | JSONL; `{timestamp, duration_seconds:float, options, llm, config_snapshot, wiki_meta, report}` (`lint/history.py:35–45`) | `save_lint_entry` — `cli/_post_compile.py:221`, `cli/cmds/quality/lint.py:210` | `load_lint_history` — `cli/cmds/quality/lint.py:344,383,433,561`, `core/reporting/log.py:55`, `lint/gap_analysis.py` | **durable / irrecoverable** | NONE — append-only vault-health audit |
| 7 | `runs.jsonl` | `RUNS_FILE` (paths.py:109; property now `runs_file` — was `runs_dir`) | JSONL; `{event:"start"|"end", run_id, cmd, ts, status?, duration_s?, exit?, error?}` (`core/reporting/runlog.py:50–59`) | `log_run_start/end` (`core/reporting/runlog.py:150`), soft-cap pruned to 1000/keep 500 | `core/reporting/log.py:51,75–130` (`knox log`) | **durable / irrecoverable** | NONE — only audit trail of command crashes/hangs |
| 8 | `memory.jsonl` | `MEMORY_LOG_FILE` (paths.py:88) | JSONL; `{timestamp, query, answer, feedback, memory_by?}` (`memory/store.py:105–117,158–170`) | `MemoryStore.add_entry` (`memory/store.py:109`) | `get_recent_entries` — `cli/cmds/knowledge/memory.py:84`, `mcp/tools/memory.py:56`; synthesis chronology | **durable / irrecoverable** | NONE — user feedback + Q/A history; zvec twin `memory.db` is rebuildable *from this file* |
| 9 | `memory.db` | `MEMORY_DB_FILE` (paths.py:87) | **NOT SQLite** despite the paths.py docstring ("SQLite memory store database", paths.py:340) — a **zvec directory** (`memory/store.py:188–227`) | `MemoryStore._ensure_collection` (`memory/store.py:174–227`) | semantic recall of past queries (`memory/store.py:132`) | **derived** | re-embed from `memory.jsonl` |
| 10 | `references.json` | `REFERENCES_FILE` (paths.py:115) | `{source_path: [Reference{id, type, title, author[], issued, container_title, …, citing_source, in_text_keys[]}]}` (`core/pipeline/references.py:41–52`; model `core/models/references.py`) | `ReferenceStore` — `compilation/two_phase.py:169`, `stages/_factory.py:128`, `cli/cmds/knowledge/references.py:126,159` | `cli/cmds/knowledge/references.py:126,159`, `lint/structural.py:791`, `mcp/tools/references.py:26,65,102` | **semi-durable** | regenerable only via LLM extraction; loss ⇒ re-extraction cost |
| 11 | `assets.json` | `ASSETS_FILE` (paths.py:113) | `{asset_id: AssetRecord{id, kind, source, data, summary, ref_count:int}}` (`core/pipeline/assets.py:35–47`) | `AssetStore` — `compilation/two_phase.py:189`, `core/pipeline/export.py:386`, `stages/_factory.py:129` | `output/vault.py:449` | **semi-durable** | regenerable; cumulative `ref_count` |
| 12 | `entity_types.json` | `ENTITY_TYPES_FILE` (paths.py:118) | flat `{concept_key: folder_name}` (`core/pipeline/entity_types.py:8–16`) | `EntityTypeRegistry` — `compilation/flows/generation.py:629,1109` | same, at generation; `output/vault.py` writes `entities/<type>/<id>.md` | **semi-durable** | regenerable, but loss ⇒ entity page folder **flicker** across runs (motivating bug, p863 roundup 2026-09-10) |
| 13 | `verification.json` | `VERIFICATION_FILE` (paths.py:95) | `{total:int, verified:int, unsupported:int, average_confidence:float, claims:[{node_id, claim, citation, verdict, confidence:float, reason}]}` (`verification.py:33–45,79–82`) | `verify_wiki` (`verification.py:79`) | `output/vault.py:571` (renders `wiki/quality.md`) | **derived** | re-run `knox verify` |
| 14 | `private.json` | `PRIVATE_FILE` (paths.py:101) | `{saved_at, pages:[{id, title}]}` (`trust.py:166–173`) | `mark_wiki_private` (`trust.py:169`) | privacy index | **derived** | recompile with privacy filter |
| 15 | `downloaded.json` | `DOWNLOADED_FILE` (paths.py:84) — **inside `cache/`** (`cache/downloaded.json`) | flat `{url: sha256hex}` (`preprocessing/downloaders/duplicate_detector.py:18–60`) | `DuplicateDetector.mark_downloaded` (`_sources_io.py:111`) | duplicate-URL check before download | **derived** | recreated as URLs re-download |
| 16 | `last_organize.json` | `ORGANIZE_REPORT_FILE` (paths.py:112) | `{dry_run:bool, categories:[…tree…], moves:[{old,new}], link_updates}` (`organize/engine.py:165–179`) | `_write_report` (`organize/engine.py:171`) | `knox organize` report display | **derived** | re-run `knox organize` (LLM-based, non-deterministic) |
| 17 | `log.md` | `LOG_FILE` (paths.py:98) | rendered Markdown report (runs + lint trend + wiki meta) | `cli/cmds/quality/log.py:61–66`, `mcp/tools/quality.py:521` | human reader | **derived** | `knox log --write` |
| 18 | `state.json` | `STATE_FILE` (paths.py:105; `VaultPaths.state_file` paths.py:401) | — | **no `src/` producer or consumer**; the **constant** `STATE_FILE` is imported by `tests/integration/test_pipeline_mock.py:33,208` and `tests/unit/test_throughput_features.py:32,112–188,458` as a throwaway filename for `TierStateStore` | **unused property** — `VaultPaths.state_file` has no `src/knox/**` caller (grep-verified), but `STATE_FILE` is NOT safely deletable (tests import it). Semantically those tests should use `COMPILE_STATE_FILE`. Leave the constant; don't reuse the property until it has a real producer. |  |  |

### 1.3 Directories under `.knox/`

| Directory | Constant | Contents & shape | Producer | Consumer | Class |
|---|---|---|---|---|---|
| `cache/` | `CACHE_SUBDIR` (paths.py:61) | parse cache + LLM cache layers | see below | see below | **derived / delete-safe** |
| `cache/<stem>-<hash>.json` | (parse cache) | one `ParsedDocument` dict: `{_source_file, _parser_fingerprint, title, author, language, page_count, metadata, elements:[…]}` (`ingest.py:379–388`; `stages/_parse.py:344–360`) | `_save_cache` `stages/_parse.py:344`; `ingest.py:178` | `_load_cache` `stages/_parse.py` | derived |
| `cache/images/` | `IMAGES_SUBDIR` (paths.py:63) | extracted figure PNGs `<stem>_img_<idx>.png` | Docling/GLM-OCR parsers `preprocessing/parsers/docling_parser.py:388–389` | renderers / VLM | derived (binaries — see §2.4) |
| `cache/extract/` | `EXTRACT_SUBDIR` (paths.py:64) | `CacheStore` layer `"extract"` = `<key>.json` | `compilation/flows/extraction.py:286,375,501`, `flows/agentic.py:213` | same, cache lookup | derived |
| `cache/chunk/` `cache/resolve/` `cache/generate/` `cache/summary/` | (CacheStore layers) | `<cache_key>.json` per layer | `stages/_parse.py:191` (chunk), `stages/_resolve.py` (resolve), `compilation/two_phase.py:321` (generate), `chunking/summarizer.py:221` (summary) | matching lookups | derived |
| `cache/encoding/` | `_ENCODING_CACHE_DIR` (text/encoding.py) | `<sha256>.utf8` normalized text copies | `normalize_to_utf8_path` `core/text/encoding.py:95` | parsers needing a file path | derived |
| `cache/gen_checkpoint/` | (literal) | `<cache_key>.json` per-batch generation checkpoints | `compilation/_checkpoints.py:141` | resume mid-generation | derived |
| `dedup/` | `DEDUP_DIR` (paths.py:110) | `fingerprints.json`: `{"_meta": {stamp, store_version}, "<slug>": {verdict, closest_match, closest_score, closest_method, embedding_score, minhash_score, embedding[], minhash[]}}` (`preprocessing/dedup/store.py:24–50`) | `_pipeline.py:335`; `preprocessing/dedup/orchestrator.py:256` | dedup matchers | **derived** (contains embeddings) |
| `vectors/` | `VECTORS_DIR` (paths.py:81) | **zvec directory store** (segments `0/`,`1/`, `idmap.0/` RocksDB, `del.*`, `manifest.*`, `LOCK`) | `ZvecWikiStore` (`rag/zvec_store.py:130–337`, `rag/__init__.py:55`) | semantic search | **derived — stays as separate optional branch** |
| `.knox/vectors.manifest.json` | (literal suffix, not a constant) | JSON list of indexed node ids (built-time id set for staleness) | `_manifest_file()` `rag/zvec_store.py:340–368` | staleness/prune check `rag/zvec_store.py:197–212` | derived — **lives NEXT to `vectors/`, not inside it** |
| `provenance/` | `PROVENANCE_DIR` (paths.py:92) | per-node `<id>.json`: `{node_id, title, citations:[marker strings]}` (`provenance.py:35–44`) | `annotate_wiki_with_provenance` (`provenance.py:39`) | `organize/moves.py:153` (rename rewrite) | derived |
| `review/` | `REVIEW_DIR` (paths.py:72) | `_state.json` `{saved_at, entries:{slug:{slug, status, title, reason, source_node_id}}}` + `<slug>.md`, `<slug>.approved.md`, `<slug>.rejected.md`, `<slug>.locked` (`review.py:77–190`) | `ReviewQueue` — `cli/_post_compile.py:138`, `cli/cmds/quality/review.py` | review CLI/MCP | **`_state.json` durable/irrecoverable** (user approve/reject + locks); markdown bodies human-editable |

---

## 2. Single-store mapping (primary proposal: SQLite at `.knox/knox.db`)

**Engine caveat:** SQLite is a recommendation, not a settled choice. The
mapping below is engine-shaped for SQLite + FTS5. §2.6 covers what changes if
zvec were the sole store instead.

Guiding rule: **one row per logical record; typed columns for anything queried
or joined; a `JSON` payload column for nested/graph-shaped content that is only
ever round-tripped as a whole.** Do not flatten the wiki; do not shred the
audit JSON.

### 2.1 Tables

| Table | Source artifact | Schema (PK in **bold**) | Rationale |
|---|---|---|---|
| `compile_state` | `compile_state.json` | **`stem`**, `source_file`, `content_hash`, `current_tier INTEGER`, `last_compile`, `search_hits INTEGER`, `last_hit`, `ingested_by` | flat, one row per source, exact-match lookups by stem → real columns |
| `wiki_snapshot` | `last_wiki.json` **and** `compiled_wiki.json` | **`kind TEXT`** (`'last'`/`'compiled'`), `nodes JSON`, `edges JSON`, `toc JSON`, `quality_score REAL`, `iteration INTEGER`, `metadata JSON`, `written_at` | the wiki is a nested graph; only ever loaded whole → JSON payload; `kind` PK lets migration reconcile the two near-duplicate snapshots into one table (or drop `compiled_wiki`) |
| `memory_nodes` | `memory.json` | **`node_id`**, `title`, `confidence REAL`, `tier`, `birth_at`, `review_at`, `review_count INTEGER`, `superseded_by NULL` | flat, queried by node + stale/decay scans → columns |
| `query_memory` | `memory.jsonl` | **`id INTEGER`** (rowid; the zvec `mem_<ts>` id retained as `mem_id TEXT UNIQUE`), `ts`, `query`, `answer`, `feedback`, `memory_by` | chronological log + feedback → columns; **FTS5 here (§2.3)** |
| `references` | `references.json` | **(`source`, `ref_id`)** composite, `type`, `title`, `authors JSON`, `issued`, `container_title`, `doi`, `url`, `raw`, `citing_source`, `in_text_keys JSON` | keyed by (source, ref); most columns scalar → typed; author/in_text_keys arrays → JSON |
| `assets` | `assets.json` | **`id`**, `kind`, `source`, `summary`, `ref_count INTEGER`, `data JSON` | id-keyed; `data` is the structured table/figure payload → JSON |
| `entity_types` | `entity_types.json` | **`concept_key`**, `folder_name` | flat map → two columns |
| `attachments` | `attachments.jsonl` | **`path`**, `source`, `ingestible INTEGER`, `embed`, `registered_at`, `deleted INTEGER` | **see §3.1 — must preserve tombstone semantics**; a per-path row with `deleted` flag is read-equivalent to latest-line-wins |
| `lint_history` | `lint_history.jsonl` | **`id INTEGER`**, `timestamp`, `duration_seconds REAL`, `options JSON`, `llm JSON`, `config_snapshot JSON`, `wiki_meta JSON`, `report JSON` | append-only; `report` is large nested → JSON payload |
| `runs` | `runs.jsonl` | **`id INTEGER`**, `event`, `run_id`, `cmd`, `ts`, `status`, `duration_s REAL`, `exit INTEGER`, `error` | append-only audit; all scalars → columns; index on `run_id` |
| `downloaded` | `downloaded.json` | **`url`**, `sha256` | flat exact-match → columns |
| `organize_reports` | `last_organize.json` | **`id INTEGER`**, `written_at`, `dry_run INTEGER`, `categories JSON`, `moves JSON`, `link_updates JSON` | tree/report nested → JSON |
| `verification` | `verification.json` | **`id INTEGER`**, `written_at`, `total INTEGER`, `verified INTEGER`, `unsupported INTEGER`, `average_confidence REAL`, `claims JSON` | summary scalars → columns; claims list → JSON |
| `private_pages` | `private.json` | **(`saved_at`, `id`)** or **`id`**, `title`, `saved_at` | tiny; one row per private page |
| `provenance` | `provenance/` | **`node_id`**, `title`, `citations JSON` | replaces N per-node files |
| `review_state` | `review/_state.json` | **`slug`**, `status`, `title`, `reason`, `source_node_id`, `saved_at` | state only; **review markdown bodies stay as files (§2.5)** |
| `dedup_fingerprints` | `dedup/fingerprints.json` | **`slug`**, `verdict`, `closest_match`, `closest_score REAL`, `closest_method`, `embedding_score REAL`, `minhash_score REAL`, `embedding JSON`, `minhash JSON`; plus a `_meta` row or a `kv` table for `{stamp, store_version}` | derived; embeddings kept as JSON (regenerable) |

`VaultPaths` gains a single `db_file` property (`knox_dir / "knox.db"`); all
store classes above reimplement `load`/`save` against it. JSONL append-only
semantics become `INSERT`; a schema-version row in a tiny `kv(key,value)` table
replaces per-file `_meta` stamps (e.g. `dedup` `store_version`).

### 2.2 JSON-in-SQLite encoding

Store nested payloads as `TEXT` with `json.dumps(..., ensure_ascii=False)`, or
`JSONB` if the SQLite build supports it. Keep the existing Python dataclass
`to_dict()/from_dict()` round-trips unchanged — only the persistence sink moves.

### 2.3 FTS5 — where it is worth it, and where it is not

| Table | FTS5? | Why |
|---|---|---|
| `query_memory` | **YES** — external-content FTS on `query`, `answer`, `feedback` | the LLM wiki-query path (§ memory #844) wants a cheap keyword fallback alongside zvec embeddings; feedback text is searchable |
| `references` | **MAYBE** — FTS on `title`, `authors`, `container_title` | bibliographic search ("find citations mentioning X"); low cost, clear win |
| `wiki_snapshot` node text | **NO by default** | `last_wiki.json` is already ~1.4 MB of nested page content; a duplicate FTS copy would bloat the single file. The primary wiki query path is the LLM reading markdown (memory #844). FTS here duplicates markdown files. |
| `lint_history`, `assets`, `compile_state`, `runs`, `downloaded`, `entity_types`, `private_pages`, `provenance`, `organize_reports`, `verification` | **NO** | exact-match or round-trip-only use; FTS adds index weight with no consumer |

### 2.4 Artifacts that CANNOT / SHOULD NOT move

| Artifact | Why it stays |
|---|---|
| `config.toml`, `rumdl.toml` | user-authored / user-editable TOML; `VAULT_MARKERS` depend on `config.toml`; rumdl preserves user edits below a marker. TOML is the trust boundary. |
| `vectors/` **and** `.knox/vectors.manifest.json` | `vectors/` is a **zvec directory store** (segments + RocksDB idmap) — not a file, and explicitly the separate optional regenerable branch. The manifest is a sidecar id-set for staleness; keep it beside the store. |
| `cache/images/`, `cache/encoding/` | binary PNGs and normalized `.utf8` copies — bulk binary blobs, no queryable structure; a DB column would bloat `knox.db` and lose the "throw the whole cache away" property |
| `cache/gen_checkpoint/`, parse-cache `<stem>-<hash>.json`, CacheStore layer dirs | content-addressed, atomic-rename, delete-safe. Optional: fold the small JSON layers (`chunk/resolve/generate/summary/extract`) into a `cache(layer, cache_key, deps JSON, payload JSON)` table; **recommend NOT** — it trades filesystem atomicity for no query benefit |
| `review/<slug>*.md` (bodies) | human-editable markdown queue; only `review/_state.json` moves |
| `memory.db` | zvec collection directory; rebuildable from `memory.jsonl` — leave it on the vector branch or regenerate |

### 2.5 `.knox/` after migration (target layout)

```text
.knox/
├── config.toml          # TOML (unchanged)
├── rumdl.toml           # TOML (unchanged)
├── knox.db              # NEW single store (all §2.1 tables)
├── cache/               # parse cache, images/, encoding/, gen_checkpoint/ (delete-safe)
├── vectors/             # zvec branch (separate, optional)
├── vectors.manifest.json
└── review/              # markdown bodies only (state moved into knox.db)
```

### 2.6 If zvec were the sole store instead

zvec is a **vector** database with schema fields + vectors, not a relational
store. Choosing it alone would force:

- **No FTS5 substitute** — the `query_memory` keyword fallback and reference
  search would be embedding-only (regresses memory #844's keyword path).
- **No transactions / unique constraints** for append-only audit files
  (`runs`, `lint_history`) or the attachment tombstone invariant — the
  `deleted`-latest-wins rule has no enforcement.
- **Awkward typed columns** — zvec fields are per-collection and vector-centric;
  `compile_state`/`entity_types`/`downloaded` are key-value state with no
  embedding, so they'd be stored as scalar zvec fields with a dummy vector, or
  kept as files anyway (defeating "single store").
- **Embedding dependency for non-embedding data** — opening the store would
  require an embedding provider to be configured.

Net: zvec-sole is only defensible if the goal is "single *directory* per
branch". For non-vector relational state, SQLite is the correct primitive; zvec
remains the embedding branch regardless.

---

## 3. Safety classification

### 3.1 Delete-safe / regenerate-on-demand

Deleting these loses nothing that cannot be rebuilt:

- **Entire `vectors/` + `vectors.manifest.json`** — regenerate by re-indexing
  (`knox compile` / rag reindex).
- **`cache/`** — parse cache, `chunk/resolve/generate/summary/extract` layers,
  `encoding/`, `gen_checkpoint/`, `images/`; `cache/downloaded.json`.
- **`dedup/fingerprints.json`** — recomputed on ingest.
- **`memory.db`** — re-embedded from `memory.jsonl`.
- **`compiled_wiki.json` / `last_wiki.json`** — recompile (but see §3.2 for the
  cost this implies for `compile_state`).
- **`provenance/`, `verification.json`, `private.json`, `last_organize.json`,
  `log.md`, `state.json`** — regenerable by their commands; `state.json` is dead.

### 3.2 Irrecoverable if deleted (or naively migrated)

| Artifact | Semantic that a naive migration breaks |
|---|---|
| **`attachments.jsonl`** | **The tombstone list.** `read_all`/`deleted_paths` use "latest line per path wins"; a `deleted:true` line is the *only* record that a user deliberately removed an attachment. An `INSERT OR REPLACE` keyed on path that drops the deleted row (or a migration that only copies non-deleted entries) makes `attach_orphan_attachments()` (`core/models/attachments.py:330–366`) silently **re-adopt the file**. Migration must carry `deleted=true` rows across and honor them. |
| **`lint_history.jsonl`** | Append-only vault-health history + the `config_snapshot`/`wiki_meta` context per run. Not recomputable — it is a time series. |
| **`runs.jsonl`** | The only audit of crashed/hung commands (a `start` with no `end`). Not recomputable. |
| **`memory.jsonl`** | User-authored `feedback` (thumbs) and Q/A answers. `memory.db` embeds *from* this; delete the JSONL and the feedback is gone even though the embeddings survive. |
| **`review/_state.json`** | User approve/reject decisions, `reason`, and `.locked` flags — human judgments, not derivable. |
| **`compile_state.json`** *(semi)* | Not user data, but deletion forces a **full recompile of every source at tier 0** — real LLM/time cost. Auto-migrate, don't clean-break. |
| **`references.json` / `assets.json` / `entity_types.json` / `memory.json`** *(semi)* | Regenerable **only via LLM extraction**, which is non-deterministic and costly; `entity_types.json` loss causes entity-page **folder flicker**; `memory.json` loss resets decay/tier clocks. Auto-migrate. |

**Biggest single risk: the attachment tombstones.** It is the one artifact
whose loss silently reverses a deliberate user action (files reappear in the
wiki) rather than merely costing recompute.

---

## 4. Migration options + risks

### 4.1 Option A — Auto-import (recommended)

On first run of the new version, read each legacy file into `knox.db`, **leave
originals in place**, and flag via `knox doctor` ("N legacy `.knox/*.json`
files present; safe to delete after verification").

- **Cost:** one-time read + insert per vault; a version marker in the `kv`
  table prevents re-import. Per-vault differences (§4.3) are handled by
  "file absent ⇒ skip table".
- **Risk:** low to data (originals untouched; rollback = drop `knox.db`).
  Residual: two sources of truth until originals are removed → a read path must
  pick the DB once imported.
- **Do NOT auto-import:** `cache/**` (delete-safe — import not worth it),
  `vectors/` + manifest (leave on the branch), `config.toml`/`rumdl.toml`
  (TOML), `review/*.md` bodies (human-editable), `memory.db` and `state.json`
  (zvec dir / dead).

### 4.2 Option B — Clean-break

Bump a storage/version marker, require `compile --force` on first run, build a
fresh `knox.db`, ignore legacy files.

- **Cost:** destroys everything derivable **and** irrecoverable on upgrade:
  tombstones, lint/runs audit, feedback, review decisions — plus forces a full
  recompile (all sources at tier 0) and LLM re-extraction for
  references/assets/entity_types. Unacceptable for the durable set.
- **Only defensible subset:** clean-break the **delete-safe branch**
  (`vectors/`, `cache/`) — those are regenerable — while still auto-importing
  the durable/semi-durable JSON state. This is a hybrid, not a true clean-break.

**Recommendation:** hybrid A + branch-reset for `vectors/`/`cache/`. Never
clean-break `attachments.jsonl`, `lint_history.jsonl`, `runs.jsonl`,
`memory.jsonl`, `review/_state.json`.

### 4.3 Existing per-vault inconsistency (already present, must be handled)

Artifacts are created **lazily**, so vaults differ. Measured on the live vault
`C:\Projects\Tech\p863-rev-eng\wiki\.knox\` (11 loose files + 3 dirs):

- **Present:** `assets.json` (2.0 MB, 632 records), `attachments.jsonl` (56 KB),
  `compile_state.json` (16 KB, 20 entries), `config.toml`, `rumdl.toml`,
  `last_wiki.json` (1.4 MB), `lint_history.jsonl` (3.5 MB), `memory.json`
  (246 KB), `runs.jsonl` (161 KB), `dedup/fingerprints.json` (884 KB),
  `vectors/` (31 files), `vectors.manifest.json` (5.4 KB, 207 ids),
  `cache/images/` (10 PNGs).
- **Absent despite being defined in `paths.py`:** `compiled_wiki.json`,
  `verification.json`, `private.json`, `downloaded.json`, `last_organize.json`,
  `references.json`, `entity_types.json`, `log.md`, `memory.jsonl`,
  `memory.db`, `state.json`, `provenance/`, `review/`.
- **`cache/` on disk is images-only** — no parse-cache `<stem>-<hash>.json`, no
  `chunk/resolve/generate/summary/extract` layers, no `encoding/`, no
  `gen_checkpoint/`. Discrete/parse caching appears disabled in this vault.
- **Curious combination:** `assets.json` has 632 records but `references.json`
  (also LLM-extraction-produced) and `entity_types.json` do **not** exist —
  so the vault stored assets without references/entity types for these sources.
  A migration must not assume one implies the other.
- **`vectors/` internals churn** (segment `1/` empty, `idmap.0/` RocksDB,
  `del.*`): treat the whole directory as opaque.
- **A stale docstring:** `paths.py:339–341` calls `memory.db` "SQLite memory
  store database" — it is a zvec directory (§1.2 #9).

### 4.4 Duplicate / dead state worth resolving during migration

- `last_wiki.json` **and** `compiled_wiki.json` are two near-identical
  `CompiledWiki` snapshots written by different paths. Pick one canonical
  (recommend collapsing into `wiki_snapshot` with `kind='last'`), or drop
  `compiled_wiki.json`.
- `state.json` / `VaultPaths.state_file` has no `src/` producer or consumer; the
  `STATE_FILE` **constant** is nonetheless imported by two test files
  (`tests/integration/test_pipeline_mock.py`, `tests/unit/test_throughput_features.py`)
  as a throwaway `TierStateStore` filename. Not deletable without those edits; leave the
  constant, wire the property only if a real producer appears. (knox-1 verified.)
- `VaultPaths.runs_dir` named a **file** (`runs.jsonl`); renamed to `runs_file` (single caller: `core/reporting/log.py:51`). The internal `RunHandle.runs_dir` field / local param name in `reporting/runlog.py` is unchanged.

---

## 5. Verification performed

- Every producer/consumer claim is a live `grep`/read against `src/knox/**`
  with the cited file:line; no claim inferred from a summary.
- On-disk inventory came from `find`/`ls` on
  `C:\Projects\Tech\p863-rev-eng\wiki\.knox\`, not from reading vault content;
  JSON shapes were sampled via `json.load` (UTF-8) on `assets.json`,
  `compile_state.json`, `vectors.manifest.json`, `memory.json`, `config.toml`.
- No `.py` file, other doc, or vault was modified. This file is the only write.
- No LLM/GPU calls; no heavy CPU loops (knox-4 owns the zvec build).
