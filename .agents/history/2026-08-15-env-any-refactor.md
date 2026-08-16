# Refactor Plan — KNOX_ env-var single source + `dict[str, Any]` → dataclasses

Created: 2026-08-14
Status: **fully implemented**. Phases 0-6 all DONE and verified.

## Implementation log (Phases 2-5, added on completion)

- **Phase 2 (C1-2, C1-3) — DONE.**
  - `LintEntry` TypedDict in `lint/history.py` (the module's own ponytail note
    said "if typed access is needed later, add a TypedDict" — the entry is a
    JSONL record whose report/options/config fields are C2/C3/C4 payloads, so a
    dataclass wrapper would add a conversion layer for nothing). All signatures
    typed; JSONL bytes unchanged.
  - `GraphEdge`/`TocEntry` TypedDicts in `core/models/wiki.py` (not dataclasses:
    `CompiledWiki.edges`/`toc` are cache-serialized — old caches hold plain
    dicts, and edges feed OKF/web JSON; a dataclass would need cache-registry
    wiring and break old-cache decoding — the exact trap fixed in D.2).
    Builders + `vault._build_related_edges` + `output/models.toc_entries` typed.
  - `TrustEvent` TypedDict (`output/models.py`, OKF §5.2), `OkfSource` TypedDict
    (§5.1) + `_reference_to_okf_source(ref: Reference)` (dropped an `Any`).
  - `ImageRef` TypedDict (`core/models/chunks.py`) wired through
    `extract_images_from_chunks` → generation → `vault._embed_images`.
- **Phase 3 (C1-4..C1-6) — DONE.**
  - Dedup: `StoreEntry` TypedDict for the persisted fingerprint entry.
    `FingerprintSet` TypedDict was tried and reverted: the per-matcher payload
    map is keyed by dynamic matcher names — TypedDict rejects non-literal keys;
    the honest type is `dict[str, CacheValue]` (matcher payloads are
    heterogeneous: embedding vectors vs serialized MinHash).
  - Parsers: `ParserMetadata` TypedDict in `_registry.py`. Everything else
    (marker/liteparse/glm/mineru/docling) audited as C4: wiki-metadata dicts,
    external-library types, MIME tables, config kwargs for vendor APIs.
  - Output: OKF sources typed (above); frontmatter stays C4.
- **Phase 4 (C1-7, C1-8) — DONE.**
  - `MemoryEntry` TypedDict (`memory/store.py`). CLI access sites now use the
    literal keys — ty requires TypedDict subscript keys to be string literals,
    and `res[MemoryField.QUERY]` (enum-typed key) is rejected; also
    f-strings use single quotes inside braces for 3.11 compat.
  - `Citation` TypedDict (`search/retriever.py`) + `structured_answer(synthesis:
    SynthesisResult)` — removed the flagged `# noqa: ANN401` `Any`.
  - `RunEvent` TypedDict (`reporting/runlog.py`) for runs.jsonl records.
  - `TemplateVariable` TypedDict (`core/templates/loader.py`).
  - Audited as keep (C2/C3/C4): mcp/tools responses (protocol payloads),
    organize (validated into CategoryTree immediately), review.py,
    acp/definitions (TOML), web helpers, cli/_shared overrides,
    _post_compile node serialization, references/assets store maps,
    cache/store registry, packs/base, zvec_store search.
- **Phase 5 (C2/C3/C4 notes) — DONE.** `ponytail:` notes added where a reader
  would otherwise "fix" honest dicts: `engine/_precedence.py` (ResolvedConfig.raw),
  `core/utils/dicts.py` (generic nested-dict utils), `core/cache/store.py`
  (registry boundary).

## Verification (final)

- `uv run pytest tests/unit -q --no-cov --tb=line` — 1671 passed, 4 skipped.
- `ruff check` — 34 (all pre-existing, unchanged from baseline).
- `ty check src/knox` — 199, **zero new diagnostics vs baseline** (2 pre-existing
  errors fixed by the D.2 conversion, none introduced).
- `scripts/check_antipatterns.py` — 8 (baseline; ENV-VAR-LITERAL clean).
- `scripts/check_field_consistency.py` — **0 violations** (the 9 pre-existing
  `MetadataField` bare-string keys were fixed post-plan: `sources.py` x2,
  `base.py` x6 → enum members; `llm/__init__.py` x1 was a lazy_loader module-name
  key (false positive) — allowlisted in the script).

## Implementation log (what changed vs. this plan)

- **B (Phase 1) — DONE.** `EnvVar` StrEnum (117 members) lives in
  `src/knox/core/env.py`, NOT `keys.py`: `knox.core.config/__init__.py` eagerly
  imports `_configs` → `paths`, so `paths.py` (loaded first) cannot import
  anything under `knox.core.config` without a circular import. `keys.py`
  re-exports `EnvVar` so all planned importers keep working; `paths.py` imports
  from `knox.core.env` directly. The `KNOX_` *prefix* strings in
  `engine/_env.py` (`removeprefix`, `env_prefix`) are structural, not var names
  — the guard exempts them.
- **Guard (B.7) — DONE.** `scripts/check_antipatterns.py` gains rule
  `ENV-VAR-LITERAL`: flags `"KNOX_..."` literals on env-access lines
  (`os.environ`, `envvar`, `validation_alias`, `_set_env`, `setenv(`, `delenv(`)
  outside `env.py`. Docstrings/help text exempt (documentation, not access).
  Test env-var literals converted to `EnvVar` too (8 sites in `test_agent.py`,
  `test_notes.py`, `test_agent_debug.py`, `test_knox_model.py`).
- **D.1 (Phase 6) — DONE.** `HttpHeader(StrEnum)` with `AUTHORIZATION`;
  comparison now `key.lower() == HttpHeader.AUTHORIZATION`. `headers` stays a
  `dict[str, str]` (see `ponytail:` note at the loop).
- **D.2 (Phase 6) — DONE, with deviations found during implementation:**
  - `Extraction.entities`/`.concepts` are now `list[Entity]`/`list[Concept]`
    (`schema.py` models). `to_dict` model_dumps, `from_dict` model_validates
    (backward compatible with existing cache JSON).
  - **Salvage removed.** `_concept_name`/`_CONCEPT_NAME_KEYS` deleted: drift keys
    (`concept`/`title`/`term`) can no longer reach `Extraction` — extraction
    validates through PydanticAI (`schema.Concept.name` is required) before
    `Extraction` is built. The salvage test was reworked to guard the remaining
    invariant (empty-name concepts skipped, no silent empty pages).
  - **Cache bug found & fixed.** The `"extract"` cache layer stored `Extraction`
    objects, but `Extraction` is not in `core/cache/store.py`'s
    `_CACHE_DATACLASSES` registry → `decode_json` raised "Unknown dataclass tag"
    → swallowed → the extraction cache was **always a miss** (silent
    re-extraction every run). Now the cache stores `result.to_dict()` and the 5
    get sites rebuild via `Extraction.from_dict(cached)` — generic cache stays
    generic, cache works.
  - Consumers converted: `flows/extraction.py` (constructors, mutation,
    synthetic-history), `flows/agentic.py`, `extraction_memory.py` (ingest +
    `to_extraction`), `lint/_remediation_gaps.py`, `flows/generation.py` debug
    warning. 65 test dict-sites converted to typed models (test_compilation's
    `ResolvedChunk(entities=...)` sites reverted — `ResolvedChunk.entities` is a
    separate `list[dict[str, Any]]`, left for C1-8).
- **Verification after D.2:** 1671 unit tests pass; ruff 34 (baseline, all
  pre-existing); ty 197 (baseline 199 — 2 pre-existing errors fixed, 0 new);
  `check_antipatterns.py` 8 (baseline); `check_field_consistency.py` 9 across 3
  files — all pre-existing `MetadataField` bare-string keys, previously hidden
  by the circular-import scan failure, now visible (separate cleanup).

## Scope

Three workstreams from one audit pass:

- **A. TODO comments** — 1 real violation, **fixed already** (see §A).
- **B. `KNOX_*` env-var strings** — 179 occurrences in `src/knox/**/*.py`, currently
  defined in 3+ places; must become a single `EnvVar` StrEnum in
  `src/knox/core/config/keys.py`.
- **C. `typing.Any` in data structures** — 269 `dict[str, Any]` occurrences across
  83 files; convert the structured payloads to dataclasses, keep the honest
  boundary types.

---

## A. TODO comments

`rg -n "TODO|FIXME|XXX|HACK" src/knox tests scripts --glob '*.py'` → 4 hits, 1 real:

| Site | Verdict |
|------|---------|
| `src/knox/llm/debug.py:92` `# TODO: make a Payload dataclass!! no typing.Any!!!` | **FIXED** — `_Payload` dataclass + `asdict()` dump (JSON shape unchanged; `tests/unit/test_agent_debug.py` green). |
| `src/knox/llm/debug.py:110`, `src/knox/core/pipeline/stages/_parse.py:260` | False positives — `\udXXX` JSON-escape comments. No action. |
| `src/knox/preprocessing/downloaders/url_resolver.py:104` | False positive — arxiv ID pattern in docstring. No action. |

---

## B. `KNOX_*` env vars — single definition

### B.1 Current state (179 occurrences, 114 distinct env names)

Definition is split across **3 places** today:

1. **`src/knox/core/config/_configs.py`** — 105× `Field(..., validation_alias="KNOX_...")`.
   The config-map aliases. Auto-derived into `ENV_MAP` by `_build_env_map()`
   (`src/knox/core/config/engine/_schema.py:32-64`). This is the de-facto primary
   definition site today.
2. **`src/knox/core/config/engine/_schema.py:66-71`** — `env_map.update({...})`
   hard-codes 5 more names: `KNOX_START_AT_TIER`, `KNOX_STOP_AT_TIER`,
   `KNOX_TIER_STATE_PATH`, `KNOX_COMPILE_CACHE_DIR`, `KNOX_RESPECT_GITIGNORE`
   (dotted keys come from `keys.py`, but the env names are literals here).
3. **Non-config env vars** (defined only at use sites):
   - `KNOX_DEBUG_LLM_DIR` — `src/knox/llm/debug.py:76` (dev-only hook, deliberately
     not a config key)
   - `KNOX_LLM_API_BASE`, `KNOX_LLM_API_KEY` — `src/knox/acp/provider_store.py:132,136`
     (litellm session credentials injected for ACP providers, not knox config)

### B.2 Hard-coded string repetitions (must become enum references)

| String | Sites |
|--------|-------|
| `"KNOX_WIKI_DIR"` literal in `envvar=` | `cli/app.py:100`, `cli/cmds/knowledge/query.py:69`, `cli/cmds/pipeline/pipeline.py:132`, `cli/cmds/quality/doctor_cmd.py:25` |
| `os.environ.get("KNOX_WIKI_DIR", ".")` | `core/paths.py:485` (dup of the config alias `PATHS_WIKI_DIR`) |
| `os.environ.get("KNOX_WIKI_DIR")` | `core/config/engine/_loading.py:117` (dup of the alias) |
| `"KNOX_LLM_DEFAULT_MODEL"`, `"KNOX_LLM_API_BASE"`, `"KNOX_LLM_API_KEY"` | `acp/provider_store.py:131-136` |
| `os.environ.get("KNOX_DEBUG_LLM_DIR")` | `llm/debug.py:76` |

### B.3 Derived occurrences (single source already, verify only)

11× `envvar=f"{APP_NAME_UPPERCASE}_..."` where `APP_NAME_UPPERCASE = "KNOX"`
(`cli/args/common.py`, `cli/args/other.py`, `cli/args/pipeline.py`). These are
construction-time constants — not duplicated literals. **Decision:** migrate them
to `EnvVar` members too, so every `envvar=` site is uniform and greppable. Delete
`APP_NAME_UPPERCASE` only if no help text still uses it (it does — keep for the
`help=` strings).

### B.4 Name collisions — NOT env vars, exclude from the enum

| Constant | Site | What it is |
|----------|------|-----------|
| `KNOX_DIR = ".knox"` | `core/paths.py:35` | filesystem directory name |
| `KNOX_CONFIG_FILE = ".knox/knox.toml"` | `core/paths.py:48` | config filename (the env var for `--config` is `KNOX_CONFIG`, a different thing) |

The `KNOX_CONFIG_FILE` path constant vs the `KNOX_CONFIG` env var is a confusing
near-collision — leave names as-is (renaming is out of scope), but the enum must
not contain the path constants.

### B.5 Docstring/help-text mentions (~45 occurrences)

Describe user-facing behavior (`"Env: KNOX_WIKI_DIR"` in Typer help). Leave as-is —
they document the *actual* env var name; making them reference the enum would
couple docs to code for no benefit.

### B.6 Target design

New `EnvVar(StrEnum)` in `src/knox/core/config/keys.py` — the **only** place a
`"KNOX_*"` literal may appear:

```python
class EnvVar(StrEnum):
    """Single source of truth for every KNOX_* environment variable name."""

    WIKI_DIR = "KNOX_WIKI_DIR"
    DEBUG_LLM_DIR = "KNOX_DEBUG_LLM_DIR"
    LLM_DEFAULT_MODEL = "KNOX_LLM_DEFAULT_MODEL"
    LLM_API_BASE = "KNOX_LLM_API_BASE"
    LLM_API_KEY = "KNOX_LLM_API_KEY"
    ...
```

All 114 names, including the 3 non-config ones (API_BASE/API_KEY/DEBUG_LLM_DIR are
still env vars; the enum documents them even though they never enter `ENV_MAP`).

Why `keys.py`, not `compilation/schema.py`: `keys.py` is already the "single source
of truth" module for key strings, imports nothing heavy (no cycle risk —
`_configs.py` and `_schema.py` both import it today). `schema.py` is the LLM JSON
output-schema module — wrong home for env vars.

Wiring (all reference the enum; no new literals):

- `_configs.py` — `validation_alias=EnvVar.LLM_DEFAULT_MODEL` etc. (StrEnum is a
  `str` subclass; pydantic accepts it). 105 sites.
- `_schema.py:66-71` — `env_map.update({...})` uses `EnvVar` members.
- CLI `envvar=` sites (B.2 + B.3) — `envvar=EnvVar.WIKI_DIR` etc.
- `core/paths.py:485`, `core/config/engine/_loading.py:117` — `EnvVar.WIKI_DIR`.
- `llm/debug.py:76`, `acp/provider_store.py:131-136` — `EnvVar.*` members.

### B.7 Regression guard

Extend `scripts/check_antipatterns.py` (the existing checker wired into
`.agents/CHECKS.md`) with a rule: no `"KNOX_..."` string literal outside
`keys.py`. This makes the "defined once" contract enforced, not aspirational.

---

## C. `dict[str, Any]` → dataclasses

### C.1 Scale

- 269 `dict[str, Any]` occurrences across **83 files** (full per-file list below).
- 188 further `Any`-parameterized containers (`list[Any]`, `dict[str, list[Any]]`, …).
- ~30 bare `Any` in signatures (`_part_to_debug_dict(part: Any)`, `_query_helpers.py:246`
  `synthesis: Any` with `# noqa: ANN401`, `agents/base.py` template machinery).

### C.2 Classification — only C1 becomes dataclasses

| Class | Meaning | Verdict |
|-------|---------|---------|
| **C1** | Structured payloads flowing between functions (fixed key set) | **Convert to dataclasses** |
| **C2** | Serialization boundaries of already-typed models (`to_dict`/`from_dict`/`model_dump`) | Keep `dict[str, Any]` — it is the honest type for JSON; optional `TypedDict` later |
| **C3** | Config-engine internals (arbitrary-depth nested config bags) | Keep — dataclass conversion is a config-layer rewrite with zero type-safety gain |
| **C4** | External format boundaries (YAML/TOML/JSON loads, frontmatter, MCP request payloads) | Keep — shapes are user-controlled, heterogeneous |

### C.3 C1 conversion targets (the actual work)

**Phase C1-1 — `llm/` (start here; partially done)**

- `llm/debug.py` — ✅ `_Payload` dataclass done (A). Remaining: `messages_to_debug_dicts`
  returns `list[dict[str, Any]]` of tagged part dicts (`type`-discriminated union of
  6 pydantic-ai part shapes). Verdict: keep as tagged dicts (serialization boundary,
  C2/C4 — a union of 6 dataclasses for a human-readable debug dump is over-build).
  `_part_to_debug_dict(part: Any)` → keep (`# noqa: ANN401` documented external type).
- `llm/postprocess.py:221` — single `dict[str, Any]`; check whether it's a payload
  dict or `model_dump()` boundary → C2.
- `llm/usage.py:1`, `llm/format.py:1`, `llm/user_providers.py:2` — audit; likely C2/C4.

**Phase C1-2 — lint history (self-contained, high value)**

- `lint/history.py` (11 sites) — lint-history entries
  (`timestamp`, `type`, `options`, `config`, `llm`, …) flow as bare dicts through
  `save_lint_entry` / `load_lint_history` / `prune_lint_history`. → `@dataclass LintEntry`
  with `to_dict()`/`from_dict()`. JSON file format must stay byte-compatible
  (entries are stored one-JSON-object-per-line; `history_to_json` keeps that).
- `lint/base.py` (4 sites) — `LintFinding.details` / `LintReport` detail dicts
  (`details: dict[str, Any] = field(default_factory=dict)` ×3). Details are
  check-specific free-form → C4 unless a stable shape emerges; audit before converting.
- `lint/runner.py:139-140` (`ref_kwargs`, `rumdl_kwargs`), `lint/structural.py:555`
  (`_find_fuzzy_duplicates`), `lint/_remediation_gaps.py` (6) — audit per site.

**Phase C1-3 — compilation graph/flow payloads**

- `compilation/graph.py` (12) — graph edges (`source`, `target`, `label`, …) and
  TOC entries built as dicts in `_build_edges`/`_build_toc`. → `@dataclass GraphEdge`,
  `@dataclass TocEntry`. Widely consumed (`output/okf_writer.py`, `output/vault.py`,
  `web/_view_helpers.py`) — convert consumers together.
- `compilation/flows/assembly.py` (9) — concept/page dicts.
- `compilation/flows/generation.py` (6) — concept/image dicts (`prepare_concepts`,
  `all_images`).
- `compilation/_two_phase_core.py` (8) — `Extraction.to_dict()/from_dict()` are C2
  (class already exists); `concept_pages()` (`list[dict[str, Any]]`) and
  `extract_images_from_chunks()` → C1.
- `compilation/wicer.py` (1), `compilation/two_phase.py` (1) — audit (likely C2,
  schema models).

**Phase C1-4 — dedup fingerprints**

- `preprocessing/dedup/orchestrator.py` (8) — `fingerprint()` returns a dict with
  stable keys (matcher name → per-matcher data); `_serialize_fp` round-trips it to
  disk. → `@dataclass Fingerprint` (or `TypedDict` if the per-matcher part is
  genuinely open). JSON on-disk format must stay compatible with existing caches
  (`dedup/store.py` reads it).
- `preprocessing/dedup/minhash_matcher.py` (2), `preprocessing/dedup/store.py` (2) — audit.

**Phase C1-5 — parser outputs**

- `preprocessing/parsers/marker_parser.py` (5) — `images` dicts, config dicts
  (`{"use_llm": ...}` → real kwargs, not a dict).
- `preprocessing/parsers/glm_ocr_parser.py` (3) — `_element_metadata` builds
  `{"label": ...}` dicts → small dataclass.
- `preprocessing/parsers/liteparse_parser.py` (3), `mineru_parser.py` (1),
  `docling_parser.py` (1) — audit (external parser JSON → C4 likely).
- `preprocessing/parsers/_registry.py` (2) — registry of parser metadata → C1
  (stable keys: name, description, …).

**Phase C1-6 — output writers**

- `output/okf_writer.py` (6) — OKF source dicts (`_reference_to_okf_source`,
  `_okf_frontmatter`) → C1 for sources; frontmatter stays C4.
- `output/vault.py` (6) — `sources`, `edges` params → C1 (reuse graph/okf types);
  `frontmatter` stays C4.
- `output/models.py` (2) — `PageOutput.frontmatter`, `toc_entries` — frontmatter C4;
  `toc_entries: list[dict[str, Any]] | None` → reuse `TocEntry`.
- `output/markdown.py:38` (frontmatter_defaults) → C4. `output/renderer.py:1`,
  `output/pages/log.py:1` — audit.

**Phase C1-7 — memory / organize / mcp**

- `memory/lifecycle.py:101-105`, `memory/store.py` (2) — `NodeState.to_dict/from_dict`
  are C2 (class exists).
- `organize/categorize.py` (2) — `_tree_from_json`/`_node_from_json` parse external
  LLM JSON → C4 (validated into `CategoryTree`/`CategoryNode` immediately).
- `mcp/tools/knowledge.py` (8), `mcp/tools/memory.py` (3), `mcp/tools/config.py` (3),
  `mcp/tools/query.py` (2) — MCP tool returns; many are `model_dump()` boundaries
  (C2). Audit per site; convert only stable internal payloads.

**Phase C1-8 — remaining C1 candidates (audit)**

- `core/templates/loader.py` (6) — `TemplateVariable` (→ C1 dataclass), `raw`
  frontmatter (C4), `_global_context` (C4 — Jinja context is open by design).
- `core/cache/store.py:101` — cache entries decode from JSON → C4.
- `cli/_shared.py:101-112` (`cli_set`/`overrides`), `cli/cmds/pipeline/pipeline.py:484`
  (`parse_cli_set`), `cli/cmds/pipeline/sources.py:141-155`, `cli/cmds/quality/doctor.py`
  (4), `cli/_post_compile.py` (2), `cli/_query_helpers.py:246` — CLI override bags are
  config-flavored → C3/C4. Audit.
- `core/pipeline/ingest.py` (2), `core/pipeline/references.py` (2),
  `core/pipeline/assets.py` (1), `core/reporting/log.py` (2), `core/reporting/runlog.py`
  (1), `search/retriever.py` (3), `rag/zvec_store.py` (1), `web/_view_helpers.py` (2),
  `review.py` (2), `acp/definitions.py:124,128` (TOML load → C4), `acp/registry.py`
  (2), `agents/base.py` (3 — template-vars machinery, C4 by design), `agents/wiki_index.py`
  (1), `agents/verification.py` (1), `packs/base.py` (1) — audit each; most are C2/C4.

### C.4 Keep-as-is (documented rationale, do not convert)

- **C2** — every `to_dict()`/`from_dict()` on an existing dataclass/Pydantic model
  (`core/pipeline/state.py:59-93`, `core/models/*.py` — 8 files, `compilation/_two_phase_core.py:71-81`,
  `memory/lifecycle.py:101-105`, `lint/runner.py:240 report_to_dict`). The JSON
  round-trip type **is** `dict[str, Any]`; hiding it behind a fake dataclass adds a
  second shape to maintain.
- **C3** — `core/config/engine/_precedence.py` (11), `_env.py` (2), `_overlay.py` (1),
  `_loading.py` (3), `config/generate.py` (4), `core/utils/dicts.py` (6 —
  `deep_merge`, `set_dot`, `flat_keys` are generic nested-dict utilities; the
  container type is the correct signature). Config values are heterogeneous by
  depth — a dataclass per level is a rewrite, not a refactor.
- **C4** — YAML/TOML/JSON/frontmatter boundaries (`core/templates/loader.py:120,457`,
  `acp/definitions.py:124`, OKF frontmatter, MCP request payloads, parser vendor JSON).

### C.5 Typing hygiene beyond dicts (same pass, low effort)

- `cli/_query_helpers.py:246` `synthesis: Any  # noqa: ANN401` — replace with the
  actual PydanticAI result type or a Protocol.
- `agents/base.py:297,495,528,556,605,614,621` — generic schema-inspection helpers;
  `type[Any]`/`Any` are the correct reflection types. Keep.

---

## D. `dict[str, str]` + hardcoded string comparisons (added per review)

Two TODOs in `src/knox/acp/provider_store.py:133-137` flag this:

```python
# TODO: headers should most likely be a data class with a well-known "Authorization"?!
# TODO: at least "authorization" must be replaced with a StrEnum value or constant expression.
for key, val in self._provider.headers.items():
    if key.lower() == "authorization":
        self._set_env("KNOX_LLM_API_KEY", val.removeprefix("Bearer ").strip())
```

### D.1 provider_store TODOs — resolution

- **TODO #2 (mandatory):** `key.lower() == "authorization"` → new `HttpHeader(StrEnum)`
  (`AUTHORIZATION = "authorization"`) in `provider_store.py`. HTTP header names are
  case-insensitive, so keep `key.lower() == HttpHeader.AUTHORIZATION`. The
  `"Bearer "` prefix strip stays a literal (well-known RFC 6750 scheme; not a
  header name).
- **TODO #1 (deliberately NOT a dataclass):** `headers: dict[str, str]` stays a map.
  Rationale: ACP `providers/set` sends an arbitrary header map (tests construct
  `headers={"x-api-key": ...}` and `{"Authorization": ...}`); only Authorization is
  consumed (rest ignored — only `KNOX_LLM_API_BASE/KEY/MODEL` envs are set). A
  dataclass with one known field plus an unknown-header bag is more code and a
  worse model than the map it replaces. The "well-known Authorization" concern is
  solved by D.1's StrEnum. Mark both TODOs resolved with a `ponytail:` note.

### D.2 `dict[str, str]` inventory — 78 occurrences / 49 files

Same C1/C2/C3/C4 lens as §C. Real C1 targets (structured records typed as maps):

| Site | Shape | Action |
|------|-------|--------|
| `compilation/_two_phase_core.py:67-68,419-420` | `Extraction.entities`/`.concepts` as `list[dict[str, str]]` | **Replace with `schema.Entity` / `schema.Concept`** — the Pydantic models already exist in `compilation/schema.py`; the loose dicts are a duplicate data model. Wire through `flows/extraction.py` + `_merge_sections` consumers. High value, low risk. |
| `output/vault.py:127` | `verified: list[dict[str, str]]` param of `okf_frontmatter` | Audit shape; if stable (`source`/`status` …) → small dataclass, else keep. |
| `llm/http_client.py:119`, `provider_store.py:43` | HTTP header maps | Keep — external/arbitrary boundary (httpx accepts `Mapping[str, str]`; consider that widening). |

Keep-as-is (honest lookup/constant/map usage — C2/C3/C4):

- `core/cache/store.py` (4), `compilation/_checkpoints.py` (4), `compilation/flows/extraction.py` (3), `compilation/flows/generation.py` (3) — dep-key maps (`gen_deps`, `extract_deps`), cache layers: id→value maps. Honest.
- `cli/_query_helpers.py:47` `_SYNTHESIS_TOKEN_KEYS` — constant table `Verbosity → config key` (values are already `keys.py` constants). Honest.
- `output/vault.py:47,72` `KIND_TO_TYPE`/`_KIND_FOLDERS`, `preprocessing/parsers/_registry.py:33,48` `KNOWN_BACKENDS`/`_BACKEND_MODULES`, `acp/provider_store.py` `_IdentityDefaultDict` — constant lookup tables. Honest.
- `organize/categorize.py:66,77`, `organize/moves.py:36,134` `mapping`, `output/okf_writer.py:135,269` + `output/pages/log.py:51` `slug_targets`/`source_titles`, `compilation/graph.py:450` `chunk_source`, `search/retriever.py:94,192` `skills`, `core/config/_configs.py:163` `role_models`, `_two_phase_core.py:557-584` `_split_sections`/`_merge_sections` (heading→text), `lint/_remediation_gaps.py:158`, `core/models/references.py:42` `to_csl()` (CSL-JSON boundary) — id→value / name→value maps. Keep.
- `cli/app.py:1`, `core/pipeline/config.py:1`, `chunking/summarizer.py:1`, `agents/base.py:1`, `packs/base.py:1`, `output/renderer.py:1` etc. — single-use contexts; audit with C2/C3/C4 default = keep.

### D.3 hardcoded string comparisons — codebase sweep

The `authorization` comparison is the flagged, mandatory fix (D.1). Full scan found
these sibling candidates (same typo-safety class, lower urgency):

| Site | Comparison | Verdict |
|------|-----------|---------|
| `agents/base.py:464` | `preset == "sliding"` | Candidate: `ContextPreset` StrEnum (check existing enums first) |
| `lint/runner.py:149` | `name == "rumdl"` | Candidate: lint-backend StrEnum |
| `lint/structural.py:200` | `field_name == "title"` | Candidate |
| `verification.py:40,69` | `verdict == "unsupported"/"supported"` | Candidate: `Verdict` StrEnum |
| `core/templates/loader.py:324` | `label in ("project", "user")` | Candidate: config-scope StrEnum |
| `core/reporting/log.py:80,94-96` | `event == "start"/"end"` | Candidate: run-event StrEnum |
| `compilation/graph.py:181`, `compilation/base.py:175` | `heading == "_root"` | Candidate: sentinel constant |
| `lint/gap_analysis.py:195` | `rule == "orphan-page"` | Candidate |
| `preprocessing/parsers/docling_parser.py:106` | `suffix.lower() == ".pdf"` | Leave — `Path.suffix` is already lowercased by pathlib; drop the `.lower()` |

Scope decision: only D.1 is in the current refactor; D.3 candidates become a
separate follow-up pass (each is a 1-line change but touches its own domain —
route through the owning subtree, check for existing enums first).

### D.4 Phase & verification

- Phase 6 (after C phases): D.1 `HttpHeader` enum + `_two_phase_core` entities/concepts
  → schema models + D.2 audit. Verify: `uv run pytest tests/unit -q --no-cov --tb=line`
  (incl. `tests/unit/test_agent.py` TestProviderConfig), `ruff check`, `ty check src/knox`,
  `uv run python scripts/check_field_consistency.py`.
- D.3: separate follow-up plan per owning subtree.

---

## E. Implementation order & verification

| Phase | Work | Verify |
|-------|------|--------|
| 0 | ✅ TODO fix (`llm/debug.py` `_Payload`) — done | `uv run pytest tests/unit/test_agent_debug.py -q` |
| 1 | B: add `EnvVar` to `keys.py` (114 names); wire `_configs.py` (105), `_schema.py` (5), CLI `envvar=` (15), `paths.py`, `_loading.py`, `debug.py`, `provider_store.py`; extend `scripts/check_antipatterns.py` | `uv run pytest tests/unit -q --no-cov --tb=line`; `ruff check`; `ty check src/knox`; `uv run python scripts/check_field_consistency.py`; `uv run python scripts/check_antipatterns.py`; smoke: `KNOX_WIKI_DIR=… uv run knox --help` |
| 2 | C1-2 lint history `LintEntry` + C1-3 graph `GraphEdge`/`TocEntry` | per-area unit tests + `tests/integration` compile smoke |
| 3 | C1-4 dedup fingerprints, C1-5 parser outputs, C1-6 output writers | dedup cache compatibility test (old JSON must load), OKF writer golden test |
| 4 | C1-7 memory/organize/mcp audit, C1-8 remaining audits | `ty check src/knox` clean |
| 5 | C2/C3/C4: no code change; add `ponytail:`-style note comments where a reader would "fix" them | docs pass |
| 6 | D.1 `HttpHeader` enum + `_two_phase_core` entities/concepts → `schema.Entity`/`schema.Concept`; D.2 audit | `uv run pytest tests/unit/test_agent.py tests/unit -q --no-cov --tb=line`; `ty check src/knox`; `check_field_consistency.py` |

Per-phase rule: a C1 conversion must not change any on-disk/JSON format
(debug dump, lint history, dedup cache, OKF output all have tests or golden files —
run them). `dict[str, Any]` in C2/C3/C4 stays; the plan's phase-5 audit documents
why at each site.

## F. Files touched (all of Part B + C1 + D1/D2)

- D.1: `src/knox/acp/provider_store.py` (+ `tests/unit/test_agent.py` if behavior asserts header extraction)
- D.2: `src/knox/compilation/_two_phase_core.py`, `compilation/flows/extraction.py`, `output/vault.py` (audit), `llm/http_client.py` (optional `Mapping` widening)
- `src/knox/core/config/keys.py` (new `EnvVar`), `_configs.py`, `engine/_schema.py`
- CLI: `cli/app.py`, `cli/args/common.py`, `cli/args/other.py`, `cli/args/pipeline.py`,
  `cli/cmds/knowledge/query.py`, `cli/cmds/pipeline/pipeline.py`,
  `cli/cmds/quality/doctor_cmd.py`
- `core/paths.py`, `core/config/engine/_loading.py`, `llm/debug.py`,
  `acp/provider_store.py`
- `scripts/check_antipatterns.py` (new env-var-literal rule)
- C1: `lint/history.py`, `lint/base.py`, `compilation/graph.py`,
  `compilation/flows/assembly.py`, `compilation/flows/generation.py`,
  `compilation/_two_phase_core.py`, `preprocessing/dedup/orchestrator.py`,
  `preprocessing/parsers/marker_parser.py`, `glm_ocr_parser.py`, `_registry.py`,
  `output/okf_writer.py`, `output/vault.py`, `output/models.py`,
  `core/templates/loader.py`, + audit-only reads for C2/C3/C4 files.
