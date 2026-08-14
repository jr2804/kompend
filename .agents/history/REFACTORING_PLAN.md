# KNOX Refactoring Plan — Super-Module Analysis

## Overview

KNOX has grown to **32,085 total lines** across **164 Python files**, of which:
- **19,966 net code lines** (actual logic)
- **1,076 comment lines** (`#` comments, counted via `tokenize`)
- **5,511 docstring lines** (triple-quoted strings, counted via `ast.get_docstring`)
- **5,532 blank lines**

Several "super-modules" and "super-functions" have evolved beyond the
target of ~350 net lines per file. This plan identifies the worst offenders
and proposes a submodule-based reorganization.

**Targets:**
- Max ~350 net lines per file (preferably ~200)
- Max 7 Python files per submodule, excluding `__init__.py` (preferably ≤5)
- Reorganize into submodules, not flat distribution
- Priority = highest net-line files first

---

## 1. Current State

### Top 20 Files by Net Lines

| Rank | File                                       | Net | Comments | Docstrings | Blanks | Total |
|------|--------------------------------------------|-----|----------|------------|--------|-------|
| 1    | `compilation/two_phase.py`                 | 867 | 60       | 116        | 112    | 1155  |
| 2    | `core/pipeline/orchestrator.py`            | 811 | 44       | 147        | 128    | 1130  |
| 3    | `compilation/_two_phase_core.py`           | 402 | 29       | 144        | 122    | 697   |
| 4    | `mcp/server.py`                            | 391 | 9        | 38         | 70     | 508   |
| 5    | `core/config/_core.py`                     | 374 | 9        | 195        | 150    | 756   |
| 6    | `cli/_pipeline_helpers.py`                 | 373 | 9        | 62         | 97     | 544   |
| 7    | `output/wiki_index.py`                     | 373 | 15       | 76         | 95     | 559   |
| 8    | `cli/cmds/query.py`                        | 369 | 12       | 44         | 68     | 493   |
| 9    | `web/server.py`                            | 363 | 2        | 84         | 37     | 486   |
| 10   | `cli/cmds/pipeline.py`                     | 343 | 6        | 32         | 41     | 422   |
| 11   | `preprocessing/dedup/orchestrator.py`      | 342 | 10       | 64         | 74     | 490   |
| 12   | `compilation/graph.py`                     | 331 | 6        | 75         | 62     | 474   |
| 13   | `lint/llm_rules.py`                        | 328 | 19       | 33         | 78     | 458   |
| 14   | `lint/remediation.py`                      | 323 | 16       | 67         | 75     | 481   |
| 15   | `compilation/wicer.py`                     | 296 | 30       | 74         | 61     | 461   |
| 16   | `preprocessing/parsers/docling_parser.py`  | 289 | 2        | 48         | 56     | 395   |
| 17   | `cli/cmds/lint.py`                         | 285 | 9        | 76         | 71     | 441   |
| 18   | `preprocessing/parsers/liteparse_parser.py`| 236 | 4        | 38         | 43     | 321   |
| 19   | `core/templates/loader.py`                 | 224 | 28       | 136        | 123    | 511   |
| 20   | `chunking/hierarchical.py`                 | 234 | 6        | 23         | 80     | 343   |

**Counting methodology:** Net = Total − Blanks − Comments − Docstrings.
Comments are `#`-style lines counted via `tokenize`. Docstrings are triple-quoted
strings counted via AST (`ast.get_docstring()`).

### Top 20 Super-Functions (>100 lines)

| Lines | File                                      | Function                    |
|-------|-------------------------------------------|-----------------------------|
| 176   | `compilation/wicer.py`                    | `compile`                   |
| 176   | `core/pipeline/ingest.py`                 | `ingest_documents`          |
| 164   | `cli/cmds/pipeline.py`                    | `compile_cmd`               |
| 146   | `compilation/two_phase.py`                | `_generate_all_pages`       |
| 128   | `preprocessing/dedup/orchestrator.py`     | `compare`                   |
| 127   | `core/pipeline/orchestrator.py`           | `run`                       |
| 125   | `acp/tools.py`                            | `build_toolset`             |
| 124   | `scaffolding/_core.py`                    | `scaffold_project`          |
| 116   | `lint/remediation.py`                     | `_remediate_coverage_gaps`  |
| 113   | `core/pipeline/orchestrator.py`           | `build_pipeline`            |
| 111   | `compilation/two_phase.py`                | `extract`                   |
| 111   | `compilation/two_phase.py`                | `compile`                   |
| 111   | `cli/cmds/pipeline.py`                    | `ingest_cmd`                |
| 100   | `cli/cmds/lint.py`                        | `lint_run`                  |
| 100   | `preprocessing/parsers/docling_parser.py` | `_extract_elements`         |
| 100   | `preprocessing/parsers/liteparse_parser.py`| `_markdown_to_elements`    |

These indicate WHERE to cut within each file — extraction targets for
sub-functions, not just which files to split.

---

## 2. Priority Targets

### Tier 1 — Critical (>600 net lines)

#### 2.1 `compilation/two_phase.py` — 867 net lines

**Problem:** `TwoPhaseCompiler` class mixes extraction, generation, and
assembly flows (18 methods) plus module-level concept-merge helpers.

**Plan — sibling `flows/` extraction (NOT package conversion):**

Keep `two_phase.py` as a module. Extract the three distinct flows into
sibling modules under `compilation/flows/`. The compiler class stays as a
coordinator that delegates to the flow modules.

```
compilation/flows/
├── __init__.py          — re-exports
├── extraction.py        — extract(), _extract_one(), _extract_incremental(),
│                          _combine_extractions(), _reduce_extractions(),
│                          _prepare_source_segments()                    (~250 lines)
├── generation.py        — generate(), _generate_all_pages(),
│                          _run_generation_with_checkpoint(),
│                          _prepare_concepts(), _gen_cache_deps()        (~200 lines)
└── assembly.py          — _assemble_wiki(), _attach_tags_to_nodes(),
                          _build_tree_texts(), _build_edges() (pure)     (~150 lines)
```

`two_phase.py` retains: `TwoPhaseCompiler.__init__`, `compile()`, static
`_batch_texts()` → **~200 lines**.

**Rationale for sibling over package:** Converting `two_phase.py` → `two_phase/`
package changes the import path for all callers and risks absorbing
`_two_phase_core.py`. Sibling extraction achieves the same net-line reduction
with minimal blast radius. Package conversion reserved for a later clean pass.

**Pattern:** Module-level functions taking `config` + state as params — not
mixins (avoids the one-product-base-class antipattern).

#### 2.2 `core/pipeline/orchestrator.py` — 811 net lines

**Problem:** `WikiPipeline` class has 41 methods spanning 5 tiers (index,
chunk, resolve, compile, output) plus caching, parsing, tag/privacy, and
wiki-assembly logic.

**Plan — tier-based `stages/` split:**

Split by pipeline tier — maps cleanly to the 5-tier domain model.

```
core/pipeline/stages/
├── __init__.py          — re-exports
├── chunk.py             — _parse_and_chunk(), _process_single_source(),
│                          _parse_source(), cache load/save for parse/chunk  (~130 lines)
├── resolve.py           — _resolve_chunks()                                (~90 lines)
├── wiki_assembly.py     — _build_wiki_hierarchy(), _apply_tags_to_tree(),
│                          _extract_source_tags(), _apply_privacy_filter()  (~130 lines)
└── caching.py           — _try_load_parsed_cache, _load_cache, _save_cache,
                          _cache_dir, _try_load_chunk_cache, _cache_key     (~150 lines)
```

`orchestrator.py` retains: `WikiPipeline` core (run, run_tier, iter_sources,
tier runners, state mgmt, path helpers) + `build_pipeline()` → **~300 lines**.

**Why tier-based over concern-based:** The tier model (index→chunk→resolve→
compile→output) is the domain vocabulary. Grouping by concern (caching/
parsing/tags) was ad-hoc; grouping by tier matches the mental model.

### Tier 2 — Important (350–600 net lines)

#### 2.3 `compilation/_two_phase_core.py` — 402 net lines

**Problem:** Internal helpers — token math, slugify, extraction dataclasses,
checkpointing, segmentation.

**Plan — extract checkpoints only:**

```
compilation/_checkpoints.py   — _gen_checkpoint_dir/path, _load/save/delete_gen_checkpoint (~80 lines)
```

`_two_phase_core.py` retains: token math, Extraction/ExtractionSet/
_BatchFailureTracker, merge/segment/image helpers → **~320 lines**.

**Rationale:** Only checkpoints are a clean separable concern. Splitting
`tokens.py` (30 lines) and `slug.py` (40 lines) into separate files is
micro-modulitis — too thin to justify.

#### 2.4 `mcp/server.py` — 391 net lines

**Problem:** MCP server with ~15 tool implementations.

**Plan — `mcp/tools/` with grouped files (NOT one-per-tool):**

One-file-per-tool (39 lines avg) violates the ≤7 files rule. Group related
tools into 4 files by domain:

```
mcp/tools/
├── __init__.py          — exports create_mcp_server
├── knowledge.py         — parse_document, compile_wiki, wiki_capture,
│                          wiki_learn, wiki_distill                        (~130 lines)
├── query.py             — query_wiki, wiki_status                         (~80 lines)
├── memory.py            — memory_add, memory_search, memory_synthesize    (~80 lines)
└── config.py            — config_get, config_set, scaffold_project        (~100 lines)
```

`mcp/server.py` retains: FastMCP app creation + tool mounting → **~100 lines**.

#### 2.5 `core/config/_core.py` — 374 net lines

**Problem:** Entire config resolution engine — merge layers, env expansion,
TOML loading, path discovery, coercion, validation.

**Plan — `engine/` with 5 files (NOT 7):**

```
core/config/engine/
├── __init__.py          — re-exports resolve_config, ResolvedConfig
├── resolve.py           — resolve_config(), _apply_layer(), _merge_project_configs() (~150 lines)
├── env.py               — _read_env_vars(), _expand_env_vars*, _classify_env_field()  (~100 lines)
├── io.py                — _load_toml(), _atomic_write_toml(), set_config_file_value(),
│                          find_project_config(), find_global_config()    (~140 lines)
└── coerce.py            — coerce_config_value(), _resolve_field_info(), validate_config_key() (~50 lines)
```

`_core.py` retains: `ConfigSource`, `ResolvedConfig`, `_build_defaults()`,
`_build_env_map()`, re-exports → **~150 lines**.

**Consolidation:** Merged the parallel agent's `merge.py` + `paths.py` into
`resolve.py`/`io.py` to stay under 5 files.

#### 2.6 `cli/_pipeline_helpers.py` — 373 net lines

**Problem:** Progress callbacks, JSON output, post-compile actions, LLM check.

**Plan — extract post-compile into sibling:**

```
cli/_post_compile.py     — _run_post_compile_lint, _cli_build_vector_index,
                           _print_compile_summary, _run_overview_synthesis,
                           _post_compile_actions, _PostCompileOptions,
                           provenance/review/private helpers               (~160 lines)
```

`_pipeline_helpers.py` retains: progress callbacks + LLM check + input
resolution → **~200 lines**.

#### 2.7 `output/wiki_index.py` — 373 net lines

**Plan — `output/pages/` split:**

```
output/pages/
├── __init__.py          — re-exports
├── index.py             — generate_index(), _node_type_counts(), _per_source_stats() (~200 lines)
└── log.py               — generate_log_entry(), prepend_log(), generate_overview() (~150 lines)
```

`wiki_index.py` → **~50 lines** (re-exports).

#### 2.8 `web/server.py` — 363 net lines

**Problem:** FastAPI routes + helpers in one file.

**Plan — flat split (NOT nested `web/api/routes/`):**

```
web/routes_pages.py      — dashboard, wiki_list, wiki_page, query_page,
                           force_graph, ingest_page, lint_page, settings_page (~200 lines)
web/routes_api.py        — api_ingest_url, api_learn, api_compile            (~50 lines)
web/_view_helpers.py     — _render_result, _source_of, _load_wiki             (~30 lines)
```

`server.py` retains: app setup + `serve()` → **~100 lines**.

**Why flat over nested:** Only 363 lines — nesting into `web/api/routes/`
with sub-sub-packages is over-structure for this size.

#### 2.9 `cli/cmds/query.py` — 369 net lines

**Plan — extract helpers to sibling:**

```
cli/_query_helpers.py    — _classify_intent, _execute_query_search,
                           _handle_adaptive_search, _handle_enhanced_search,
                           _synthesize_and_output, _build_enhanced_retriever,
                           _save_query_memory, encoding helpers             (~200 lines)
```

`query.py` retains: `query_cmd` arg parsing + dispatch → **~150 lines**.

### Tier 3 — Borderline (250–350 net lines)

| File                                      | Net  | Decision |
|-------------------------------------------|------|----------|
| `cli/_sources.py`                         | 318  | **Split** → `cli/_sources_io.py` (archive/download/copy) |
| `preprocessing/dedup/orchestrator.py`     | 342  | **Split** — `compare` (128 lines) → `preprocessing/dedup/compare.py` |
| `compilation/graph.py`                    | 331  | **Split** — `_build_concept_pages` → `compilation/_graph_pages.py` |
| `lint/remediation.py`                     | 323  | **Split** — `_remediate_coverage_gaps` (116 lines) → `lint/_remediation_gaps.py` |
| `compilation/wicer.py`                    | 296  | **Split** — `compile` (176 lines) → `compilation/_wicer_loop.py` |
| `cli/cmds/lint.py`                        | 285  | Review — mostly rule definitions |
| `preprocessing/parsers/docling_parser.py` | 289  | Review — parser-specific |
| `core/templates/loader.py`                | 224  | OK — high docstring ratio (good docs) |
| `preprocessing/parsers/liteparse_parser.py`| 236 | OK — under threshold |
| `chunking/hierarchical.py`                | 234  | OK — under threshold |
| `organize/categorize.py`                  | 234  | OK — under threshold |

### Deferred — `cli/cmds/` reorganization

`cli/cmds/` has 19 files (over the 7 limit). **Do NOT split files further**
(e.g., splitting `pipeline.py` into `compile.py` + `ingest.py` makes it 20).
The correct fix is grouping into domain subdirs:

```
cli/cmds/pipeline/   — compile, index, status, config
cli/cmds/quality/    — doctor, lint, review, log
cli/cmds/knowledge/  — query, capture, organize, memory, sources, pack
```

Defer until the file-splitting phases are stable. Flat layout is defensible
while files are small and self-contained.

---

## 3. Refactoring Order

Dependency-ordered to avoid breaking changes. Each phase: extract → run full
test suite → commit independently.

### Phase 1 — Independent splits (no cross-module deps)

| # | Target | Result | Est. |
|---|--------|--------|------|
| 1 | `output/wiki_index.py` → `output/pages/` | 373 → 2 files × ~175 | 2h |
| 2 | `cli/_sources.py` → `cli/_sources_io.py` | 318 → 2 files × ~160 | 2h |
| 3 | `compilation/_two_phase_core.py` → `_checkpoints.py` | 402 → 2 files × ~200 | 2h |

### Phase 2 — Module splits (depend on Phase 1)

| # | Target | Result | Est. |
|---|--------|--------|------|
| 4 | `compilation/two_phase.py` → `compilation/flows/` | 867 → 3 + coordinator ~200 | 6h |
| 5 | `web/server.py` → `routes_pages.py` + `routes_api.py` | 363 → 3 + server ~100 | 3h |
| 6 | `mcp/server.py` → `mcp/tools/` | 391 → 4 + server ~100 | 4h |

### Phase 3 — CLI splits (depend on Phase 2)

| # | Target | Result | Est. |
|---|--------|--------|------|
| 7 | `cli/_pipeline_helpers.py` → `_post_compile.py` | 373 → 2 files | 3h |
| 8 | `cli/cmds/query.py` → `_query_helpers.py` | 369 → 2 files | 3h |

### Phase 4 — Core splits (depend on all above)

| # | Target | Result | Est. |
|---|--------|--------|------|
| 9  | `core/config/_core.py` → `engine/` | 374 → 5 + core ~150 | 4h |
| 10 | `core/pipeline/orchestrator.py` → `stages/` | 811 → 4 + orchestrator ~300 | 6h |

### Phase 5 — Cleanup

| # | Target | Est. |
|---|--------|------|
| 11 | `compilation/graph.py` (331) → `_graph_pages.py` | 2h |
| 12 | `lint/remediation.py` (323) → `_remediation_gaps.py` | 2h |
| 13 | `compilation/wicer.py` (296) → `_wicer_loop.py` | 2h |
| 14 | `preprocessing/dedup/orchestrator.py` (342) → `compare.py` | 2h |
| 15 | Import compatibility sweep + docstring audit | 3h |

---

## 4. Target State

| File | Current Net | Target Net | Action |
|------|-------------|------------|--------|
| `compilation/two_phase.py` | 867 | ~200 | Sibling `flows/` extraction |
| `compilation/flows/extraction.py` | — | ~250 | New |
| `compilation/flows/generation.py` | — | ~200 | New |
| `compilation/flows/assembly.py` | — | ~150 | New |
| `core/pipeline/orchestrator.py` | 811 | ~300 | Tier-based `stages/` split |
| `compilation/_two_phase_core.py` | 402 | ~320 | Extract `_checkpoints.py` |
| `mcp/server.py` | 391 | ~100 | `tools/` (4 grouped files) |
| `core/config/_core.py` | 374 | ~150 | `engine/` (5 files) |
| `cli/_pipeline_helpers.py` | 373 | ~200 | Extract `_post_compile.py` |
| `output/wiki_index.py` | 373 | ~50 | `pages/` split |
| `cli/cmds/query.py` | 369 | ~150 | Extract `_query_helpers.py` |
| `web/server.py` | 363 | ~100 | Flat route split |
| `cli/cmds/pipeline.py` | 343 | — | Deferred to cli/cmds/ reorg |
| `compilation/graph.py` | 331 | ~200 | Extract `_graph_pages.py` |
| `lint/remediation.py` | 323 | ~200 | Extract `_remediation_gaps.py` |
| `compilation/wicer.py` | 296 | ~150 | Extract `_wicer_loop.py` |

**Goals:**
- No file exceeds 350 net lines
- Submodules contain ≤5 files each (all new submodules stay at 5)
- Import compatibility maintained via re-exports where needed
- Docstrings preserved

---

## 5. Estimated Effort

| Phase | Files | Est. Hours |
|-------|-------|------------|
| 1 — Independent splits | 3 | 6 |
| 2 — Module splits | 3 | 13 |
| 3 — CLI splits | 2 | 6 |
| 4 — Core splits | 2 | 10 |
| 5 — Cleanup | 5 | 11 |
| **Total** | **15 targets** | **~46h / ~6 sessions** |

---

## 6. Comment and Documentation Analysis

| File | Comments | Docstrings | Net | Comment % | Docstring % |
|------|----------|------------|-----|-----------|-------------|
| `core/templates/loader.py` | 28 | 136 | 224 | 12.5% | 60.7% |
| `compilation/wicer.py` | 30 | 74 | 296 | 10.1% | 25.0% |
| `compilation/_two_phase_core.py` | 29 | 144 | 402 | 7.2% | 35.8% |
| `compilation/two_phase.py` | 60 | 116 | 867 | 6.9% | 13.4% |
| `core/pipeline/orchestrator.py` | 44 | 147 | 811 | 5.4% | 18.1% |

**Findings:**
- All under 13% comment ratio — no redundant-comment debt
- `core/config/_core.py` has 52% docstring ratio — this is good API docs, not bloat
- `core/templates/loader.py` has substantive docstrings (template precedence, validation) — preserve, optionally expand to `docs/`
- `compilation/wicer.py` comments explain the CEGAR algorithm (arXiv:2605.07068v1) — preserve and link to paper

**No action needed** on moving comments to docs — the codebase is lean on prose.

---

## 7. Risks and Mitigations

### Import Breakage

**Mitigation:**
- Use re-export pattern: keep old paths working via `__init__.py` re-exports
- Run `rg "from knox.X import Y"` before and after each split
- Prefer sibling extraction over package conversion (lower blast radius)

### Circular Dependencies

**Mitigation:**
- Analyze import graph with `codegraph_callers`/`codegraph_callees` before splitting
- Keep shared utilities in leaf modules (no outward imports)
- Use `from __future__ import annotations` for type hints

### Test Coverage

**Mitigation:**
- Run full test suite before and after each phase
- Each phase is independently committable and revertable
- Run `check_field_consistency.py` + `check_antipatterns.py` after each phase
