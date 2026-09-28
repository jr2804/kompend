# Plan — Markdown-Only Vault: the wiki is the product, `.knox/` is derived state

Created: 2026-08-22
Status: **implemented** — WP1–WP7 all landed (2026-08-22). Remaining polish: none planned.

## 1. Problem

Using a knox vault from another project (git repo, `.knox/` excluded — it holds
API keys / provider config) exposed a design violation:

1. **knox itself couldn't query** — `wiki_query` (MCP) requires
   `.knox/last_wiki.json`; without it: *"No compiled wiki found. Run
   'wiki_compile' first."* Expected (no config), but it also means **query
   correctness depends on derived cache state**, not on the wiki pages.
2. **External agents pointed at `index.md` got poor/empty answers** — the
   catalog is one-liners, and several kinds of query-relevant information
   exist *only* inside `.knox/` (assets, verification, provenance detail,
   per-source reference mapping, extracted images).

That is a **two-branch** architecture: "knox present" (rich) vs "knox absent"
(degraded). It contradicts the LLM-wiki principle: *the wiki is represented by
its generated pages alone*. Everything an LLM needs must be in markdown;
`.knox/` must be purely derived state (cache, indexes, vectors, config).

## 2. Current-state audit (evidence)

### 2.1 What lives only in `.knox/` and whether it is knowledge

| Artifact | Content | Knowledge? | Exported to markdown today |
| --- | --- | --- | --- |
| `references.json` | full CSL `Reference`s per source | **yes** | ✅ `wiki/references.md` + in-text citation rewrite (`rewrite_source_references`) — but page is unlinked from `index.md`, per-source grouping lost |
| `assets.json` | full separated tables/figures (`AssetRecord.data`), summaries | **yes** | ⚠️ footnote summaries only (`[^tab-xxx]: summary`); full table/figure data never leaves `.knox/` |
| `cache/images/` | parser-extracted images embedded in wiki pages | **yes** | ❌ wiki pages embed relative paths `.knox/cache/images/…` (relativized to base_dir in `ingest.relativize_image_paths`) → **broken links** in any vault without `.knox/` |
| `verification.json` | WiCER claim-verification results | **yes (trust)** | ❌ not exported |
| `provenance/` | evidence traces per generated claim | partially | ⚠️ structured §5.1 `sources` objects in page frontmatter; raw traces stay internal |
| `memory.db` / `memory.json` / `memory.jsonl` | query memory + node-lifecycle states (stale/superseded/promotion) | no | n/a — session/pipeline state (see §4.3) |
| `last_wiki.json` / `compiled_wiki.json` | serialized `CompiledWiki` (edges, toc, quality) | derived | ⚠️ `load_wiki_from_files` reconstructs from markdown — **CLI only** |
| `search.db` | FTS5 index | derived | n/a — rebuildable |
| `vectors/` | embeddings | derived | n/a — stays (per decision below) |
| `review/`, `private.json`, `compile_state.json`, `state.json`, `runs.jsonl`, `lint_history.jsonl`, `last_organize.json`, `downloaded.json`, `config.toml` | pipeline state, secrets, privacy manifest | no | n/a — stays in `.knox/` |

### 2.2 The two-branch problem in code

- `src/knox/cli/cmds/knowledge/query.py`: `load_cached_wiki()` first, then
  portable fallback `load_wiki_from_files()` with a "Portable mode" notice.
  Two behaviors, cache wins — markdown is not the source of truth.
- `src/knox/mcp/tools/query.py::query_wiki`: **only** `load_cached_wiki()`,
  no fallback → hard error without `.knox/`. (Same pattern in
  `mcp/tools/pipeline.py`, `mcp/tools/quality.py`.)
- `load_wiki_from_files()` (`src/knox/core/models/wiki.py`) already exists and
  is tested (`tests/unit/test_portable_wiki.py`) — the primitive we need is
  there; it is just not the primary path.

## 3. Target state

```text
base_dir/
├── index.md            # vault-root hub → wiki/index.md (+ sources/, attachments/)
├── sources/            # inputs (untouched, user-owned; organize may rearrange)
├── attachments/        # per-source media incl. copy-on-embed images for wiki pages
├── wiki/
│   ├── index.md        # portable catalog: links EVERYTHING (pages, references, summaries, quality)
│   ├── overview.md, log.md, references.md
│   ├── concepts/…      # wiki pages (unchanged layout)
│   ├── summaries/      # NEW — one summary page per ingested source
│   └── quality.md      # NEW — verification summary (WiCER, per-page table)
├── exported/           # NEW — full-source markdown exports (reference only, on demand)
└── .knox/              # derived state ONLY: cache/ (incl. images/), assets.json,
                        # vectors/, search.db, config.toml, pipeline state,
                        # memory, provenance, review, quality_score
```

**Invariant (the one-line acceptance test):** delete `.knox/` → every fact an
LLM needs is still reachable by following markdown links from `wiki/index.md`,
and `knox query` / `wiki_query` return the same quality of results.

## 4. Decisions (proposed — discuss)

### 4.1 Exported feature pages live *inside* `wiki/`, not in new top-level dirs

- `wiki/` is the product; the hub-and-spoke contract is "follow links from
  `wiki/index.md` and you have everything". New top-level dirs (e.g. `refs/`,
  `assets/`) would add a second root an agent must know about.
- Top level stays small and purpose-named: `index.md`, `sources/` (input,
  user-owned), `attachments/` (durable media), `wiki/` (the product),
  `exported/` (reference dumps, appears only after an export), `.knox/`
  (derived state). Media (binary) goes to `attachments/`; *generated
  knowledge* goes to `wiki/`; *reference dumps* go to `exported/`.
- Alternative rejected: `assets/` next to `sources/` — splits generated
  knowledge across roots, breaks the "wiki = pages only" principle.

### 4.2 Markdown is the source of truth for query

- `load_wiki_from_files()` becomes the **primary** loader everywhere
  (CLI + MCP). `last_wiki.json` is downgraded to a pure accelerator:
  used only as a *validated* fast path (stamp it with a content hash of the
  wiki tree; stale cache → silently fall back to files, rebuild).
- No behavioral difference between "knox present" and "knox absent" for
  query/read paths. Compile/lint/maintenance paths may use cache freely.

### 4.2a Compiled-wiki field disposition (review comment 1)

What `CompiledWiki` carries beyond pages, and where it belongs:

| Field | Disposition | Evidence |
|---|---|---|
| `nodes` | markdown pages — already the source of truth | `load_wiki_from_files` reads frontmatter + body |
| `edges` | **derivable from markdown** — pages' body links + related-sections already encode them | `load_wiki_from_files` rebuilds edges from `[title](concepts/x.md)` / `[[x]]` links. Only edge *type* and *confidence* are cache-only; query uses edges solely in graph-expansion mode (`_graph_expansion_results`, decay 0.5), which works fine with untyped edges. Wiki pages already render related links via `_build_related_edges` |
| `toc` | **not exported** — `wiki/index.md` *is* the TOC | consumers are typst/json writers, provenance export, and `toc_count` in status — all compile-time derivable from page titles/frontmatter; nothing in retrieval reads it |
| `quality_score` | `.knox/` only (status/quality tools) | `compute_quality_score(nodes, edges)` — pure function of nodes+edges, recomputable on demand |

`HybridRetriever` despite its name is **purely vector-based** (zvec over
`.knox/vectors/`); with `vector_store=None` it returns `[]`. The actual
markdown fallback is `KeywordRetriever` (title tokens 3x, content 1x) —
CLI-only today. So MCP `query_wiki` without vectors returns *nothing*
even when `last_wiki.json` exists — that is the "returned nothing" symptom.
Query functionality itself needs only nodes (+ edges): both come from
markdown. Vectors are the optional recall booster (§4.4).

### 4.3 Memory stays in `.knox/`

- `memory/` holds query history and node-lifecycle states — session state,
  not source knowledge. Nothing query-relevant is lost.
- If distilled insights should become wiki knowledge, the existing
  `distill` → `sources/_distilled/` → compile flow is the intended path.

### 4.4 Vectors stay in `.knox/` (user decision, confirmed)

- RAG/vector branch is an optional knox-only accelerator. Without it, query
  degrades to keyword/FTS over the same markdown — same corpus, same answers,
  lower recall on paraphrase. Document this degradation, don't hide it.

### 4.5 Assets and parse-cache images stay in `.knox/` (review comments 2 & 4)

- `assets.json` is a compilation/embedding working store — full tables and
  figures are **not** exported as wiki pages. Wiki pages keep the footnote
  summaries (`¶ tab-xxx` → `[^tab-xxx]: summary`).
- Full-fidelity asset markdown goes through the existing **`knox sources
  export`** (annotations `inline` renders assets as H2 sections + images +
  enrichments). Default target becomes `<root>/exported/<stem>.md` — a
  top-level `exported/` dir, **not** under `wiki/`: exports are reference
  material for lookup, not wiki pages. The `OUTPUT` CLI argument becomes
  optional (omit → default target); passing it (or an explicit
  `--output-file`) exports to any arbitrary path. On demand, not default-on.
- Parse-cache images stay in `.knox/cache/images/` — they feed the VLM
  enrichment pass and the source export, not the wiki.
- Only portability fix (WP3, revised): at vault write, any image actually
  embedded into a wiki page (`_embed_images`) is **copied into the existing
  `attachments/<slug>/` per-source layout** and the embed link rewritten.
  `attachments/` remains *the* place for durable media; pages never
  reference `.knox/`. No new `_extracted/` dir.

## 5. Work packages

### WP1 — Make markdown-first query the only branch

1. `query_wiki` (MCP) and all `load_cached_wiki()` consumers in
   `mcp/tools/` switch to `load_wiki_from_files()` primary (or a shared
   `load_wiki(base_dir)` helper in `core/models/wiki.py` that encapsulates
   cache-validation + fallback).
2. **MCP keyword fallback**: `query_wiki` currently builds
   `HybridRetriever(wiki, vector_store=None)` → always `[]`. Port the CLI
   dispatch (`_execute_search` in `cli/cmds/knowledge/query.py`):
   no vector store → `KeywordRetriever(wiki)`. Extract that dispatch into a
   shared helper so CLI and MCP cannot diverge again.
3. CLI `query.py`: drop the two-branch print; same shared loader.
4. Cache validation: `last_wiki.json` gains a wiki-tree fingerprint
   (hash of page paths + mtimes, or content hash); mismatch → rebuild from
   files. Cache never wins over changed files.
5. Acceptance: with `.knox/` deleted, `knox query` and `wiki_query` return
   results identical to cached mode on a test vault (keyword mode).

### WP2 — Per-source markdown surfaces

**Summaries** — one generated page per ingested source at
`wiki/summaries/<stem>.md` (review: *not* keyed to or placed near
`sources/` — that folder is user-owned and `organize` may rearrange it;
summaries must survive re-organization and live with the product):

1. frontmatter: `type: Source`, source path, tags, stats (chunks, nodes fed)
2. body: one-paragraph digest (reuse chunk summaries from the parse cache —
   no new LLM calls), **asset inventory** (table/figure ids + one-line
   summaries — the per-source asset knowledge that today lives only in
   `assets.json`), **citations extracted from this source**, backlinks:
   concepts this source feeds (inverse of concept pages' `source_files`).
3. Full-source export is linked from the summary page only if an export
   exists — no default-on export; plain text search uses keyword/embeddings,
   not full dumps.
4. `wiki/index.md` gains a Summaries section.

**Export defaults** — `knox sources export`:

1. `OUTPUT` argument becomes optional; omitted → `<root>/exported/<stem>.md`
   (dir created lazily; `exported/` is reference material, never linked as
   wiki content beyond the per-source summary pointer).
2. Explicit `OUTPUT` (any arbitrary path) keeps working unchanged.
3. `exported/` in `.gitignore` template? — decide: no, it's deliberate
   reference material; document instead.

### WP3 — Copy-on-embed: wiki pages never reference `.knox/`

Revised per review comment 4 — no `_extracted/` move, images cache stays.

1. At vault write (`_embed_images`): if an `ImageRef.path` resolves under
   `.knox/`, copy the file to `attachments/<source-slug>/` and rewrite the
   embed to a page-relative link (`../attachments/<slug>/…`).
2. Source slug from `attachment_slug(source)` (existing helper); images
   keep their filename (`<stem>_img_<n>.png`).
3. Guard: extend the structural lint (WP-global) to fail on any page link
   or embed resolving into `.knox/`.
4. Acceptance: new compile on a PDF-containing corpus → images render in a
   plain markdown viewer with `.knox/` deleted; `.knox/cache/images/` still
   present (cache untouched, regenerable).

### WP4 — References page completeness

1. Drop sequence numbering (review comment 3 — decided); cite by stable
   content-hash id everywhere (in-text rewrite + bibliography anchors).
2. `wiki/references.md` stays a flat alphabetical bibliography keyed by
   hash id; the **per-source citation mapping** moves to the per-source
   summary pages (WP2) — that's where grouping belongs.
3. Link `references.md` from `wiki/index.md` (and vault-root `index.md`).

### WP5 — Verification / quality export

1. `wiki/quality.md`: WiCER verification summary — **per-page table**
   (page, claims checked, pass/fail, score; review comment — decided)
   - overall stats + compile stamp. Link from `wiki/index.md`.
2. Per-page: verification outcome goes into OKF frontmatter extension key
   (e.g. `verified_claims`, `quality_score`) — cheap, machine-readable,
   no page-body noise.
3. `provenance/` traces stay internal (verbosity); §5.1 `sources` frontmatter
   objects remain the per-page provenance surface. If a consumer needs raw
   traces, `knox provenance export` can come later — YAGNI now.

### WP6 — Index as complete catalog

1. `wiki/index.md` (and vault-root `index.md`) must link: all page groups,
   `overview.md`, `log.md`, `references.md`, `summaries/`, `quality.md`.
2. Each entry keeps the one-line description (catalog style) — that's by
   design; the fix for "index alone answers poorly" is complete *linkage*,
   not fatter index. Agents follow links.

### WP7 — Docs, scaffolding, skill

1. `docs/guides/wiki-structure.md`: new layout tree, markdown-only invariant,
   portability guarantee, vector-degradation note, `exported/` semantics.
2. `docs/concepts/basic-principles.md`: state the principle —
   "the wiki is complete in its generated pages; `.knox/` is derived state".
3. `.agents/FILES.md` + `src/knox/core/paths.py` docstring: updated layout.
4. `knox.toml.example` if any new config keys emerge (avoid: no new keys
   unless a toggle is genuinely needed — exports are unconditional).
5. **Scaffolding** (`src/knox/scaffolding/templates/AGENTS.md.j2`): new
   section "No knox available? Navigate the wiki yourself" — entry point
   `wiki/index.md` (portable catalog), then `references.md`, `summaries/`,
   `quality.md`, `log.md`; note that everything reachable from the index is
   the complete knowledge surface; `.knox/` never required for reading.
6. **Skill** (`skills/knox/SKILL.md` — the agent-facing skill installed by
   knox clients via MCP/CLI): update alongside the docs —
   - Directory Layout section → new tree (`wiki/summaries/`, `quality.md`,
     `exported/`, `.knox/` marked *derived state, never required for
     reading*); it also drifts today (`pages/`, `wiki.md`-as-main-output,
     `sources/_memory/` are stale).
   - Query & Retrieval section → markdown-first semantics: `knox query` /
     `wiki_query` work from the wiki files alone; vectors only boost recall.
   - New subsection "Vault without knox" — for agents that consume a vault
     with no knox installed: start at `wiki/index.md` (portable catalog),
     follow links; `references.md`, `summaries/`, `quality.md`, `log.md`;
     everything reachable from the index is the complete knowledge surface.

## 6. Order & dependencies

```text
WP1 (query single-branch + keyword fallback) — independent, highest value, first
WP3 (copy-on-embed images)          — independent, small, fixes broken media
WP2 (per-source summary pages)      — after WP4 decides citation id format
WP4 (references by hash id)         — independent
WP5 (quality page)                  — independent
WP6 (index catalog)                 — last, after 2/4/5 exist to link
WP7 (docs)                          — with each WP, final pass at end
```

## 7. Verification

- Per WP acceptance criteria above.
- Global: integration test — compile fixture corpus → snapshot wiki tree →
  delete `.knox/` → (a) `load_wiki_from_files` reconstructs equal nodes;
  (b) `knox query` keyword-mode results equal; (c) no `[[wikilinks]]` or
  image/citation links resolve into `.knox/`.
- New lint rule (structural lint already exists in `lint/structural.py`):
  fail if any wiki page references a path under `.knox/`.
- Existing: `uv run pytest tests/unit -x`, `ruff check`, `ty check src/knox`,
  `check_field_consistency.py`, mkdocs `--strict`.

## 8. Resolved risks / open questions

1. **`load_wiki_from_files` fidelity — resolved** (§4.2a): `edges` derive
   from page links (types/confidence cache-only, cosmetic for expansion);
   `toc` = `wiki/index.md` itself, not needed by retrieval; `quality_score`
   recomputable, stays in `.knox/`.
2. **Per-source summary page cost** — one small page per source; digest can
   reuse the chunk-level summaries already in the parse cache (no new LLM
   calls needed unless we want a richer digest — start cache-only).
3. ~~References numbering~~ — dropped (review comment 3).
4. ~~`attachments/_extracted/`~~ — rejected (review comment 4); copy-on-embed
   into existing `attachments/<slug>/` instead.
5. ~~Old vaults~~ — confirmed: no migration, recompile regenerates
   (review comment 5).
6. ~~`sources export` default mode~~ — resolved: default target
   `<root>/exported/<stem>.md`, annotations `inline` when writing into the
   vault; arbitrary-path output via explicit `OUTPUT`.
7. ~~quality.md granularity~~ — resolved: per-page table + overall stats.
8. **Open: summary-page keying** — keyed by source *stem*; if `organize`
   moves/renames a source, next compile regenerates summaries (stale ones
   pruned like wiki pages). Collision risk for same-stem sources — use
   `attachment_slug` (path-derived, collision-free) as the filename.

User Comments:

1. `quality_score` -> .knox folder, `toc` -> wiki, if needed for navigation (?) in addition to index.md? Please explain `edges` and `HybridRetriever` more in detail. Are they necessary for query functionality, or can they be derived from the markdown pages?
2. assets should primarily only be used for compilation process and embedding -> can remain in .knox folder. We recently implemented also the "full markdown export" for a source - this should contain all assets as well (as H2 sections + images + enrichments). Not sure if we should in general allow the "full source as markdown"-feature by default; for plain text search, we should rather use embeddings. For a quick per-source overview, we can generate a summary pages for each source?
3. Drop numbering.
4. Not sure if we should move images to a new `_extracted` folder. These should actually only be used during compilation (extracted enrichments/vlm pass) and when exporting the "full source as markdown" feature. The "attachments" folder should remain the main place for all attachments (per-source). Seems to challenge WP3 in general?
5. Correct, no legacy detection or migration needed.
