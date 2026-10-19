# AGENTS.md — History

Past design decisions, known issues, and "why we banned X." Optional background
reading; not required for day-to-day work.

## General History

### Pattern: Design Decision Log

Record significant decisions that shaped the codebase. Each entry explains the
context, the decision, and where it landed. This prevents future contributors
from re-litigating settled questions.

### Rules

- **Newest first**, **one entry per decision**, 2–6 lines: context, decision,
  commit. Detail belongs in `.agents/history/` — link it.
- **Budgeted like every `.agents/` file (≤120 lines).** Over budget → archive the
  oldest entries to `.agents/history/`, leaving a pointer line.
- Move resolved entries to a "Resolved" subsection rather than deleting them.

### Extension

| Who | When | What |
|-----|------|------|
| Any contributor | Making a notable design decision | Add entry at top |
| Project maintainer | Resolving a known violation | Move to Resolved subsection |
| Project maintainer | Deprecating a tool/pattern | Add rationale entry |

---

## Project-specific History — kompend

### Archived plans — full detail in `.agents/history/`

| Archived | Slug | What landed |
| -------- | ---- | ----------- |
| 2026-09-16 | `context-window-provider-overrides` | `[providers.<name>].context_window` now beats auto-discovery (step 1 of `discover_context_window`); `kompend doctor` stops reporting override-only blocks of *known* providers as `missing base_url` while still erroring on unknown ids |
| 2026-09-24 | `kompend-0iv2-audit` | Read-only audit resolving a tracker-vs-code contradiction from the code; the reusable part is the method — dynamic-access sweep plus `git log -S "<Class>"` |
| 2026-09-24 | `kompend-storage-migration` | `.kompend/` consolidated into one SQLite file (`core/store/`); clean cut, no migration code — `61147c5f`, `f6aee4c0`, `8aaf2eca` |
| 2026-08-22 | `markdown-only-vault` | The wiki is represented by its generated markdown alone; `.kompend/` is purely derived — `5d493119`, `51ea1111` |
| 2026-08-16 | `source-export-md`, `category-extraction`, `rumdl-format-wiring` | `kompend sources export` from the parse cache; configurable extraction categories; rumdl wiring for every markdown writer |
| 2026-08-15 | `env-any-refactor`, `rlm-integration` | Typed config access replacing `Any`; the optional RLM code-exploration capability |
| 2026-08-14 | `migration-plan`, `refactoring-plan` | `summarization-pydantic-ai` context management (`edb4d49`); super-module splits into `stages/`, `engine/`, `tools/`, `flows/`, `pages/` |

### Why text-in-code is banned

Prompts embedded in Python were undiscoverable, unversionable, and invisible to
non-developers. All prompt text now lives in Jinja2 templates or pack TOML. The
remaining legacy violations are tracked in beads — `bd list`, not a table here.

### Why litellm was removed

`c8055c0`. After moving to pydantic-ai's native `infer_model()` plus the
models.dev catalog, litellm was 40+ transitive dependencies and added latency
around a unified `build_model()` path. Removed.

### Why anthropic is pinned below 1.0 (2026-10-18)

pydantic-ai 2.31.1's Anthropic provider hands the SDK an `httpx.AsyncClient`,
while `anthropic>=1.0` moved to `httpx2` and rejects it at construction
(`TypeError: Invalid http_client argument`), and pydantic-ai-slim 2.31.1 still
declares `anthropic>=0.108.0` with no upper bound. `[tool.uv]
constraint-dependencies = ["anthropic<1.0"]` holds the SDK on the 0.x line
(`httpx`, resolves to 0.125.0); drop it when pydantic-ai is upgraded (newer
pydantic-ai already requires `anthropic>=1.12.1`).

### Why heavy imports are lazy-loaded (SPEC-0001)

`kompend --help` took ~80s because `register_all_commands()` imported every command
module, each transitively pulling docling/torch/pydantic-ai/MCP. `lazy.attach()`
in five package `__init__.py` files, plus the `get_backend()` factory for
function-body-only classes, took it to 11s (−86%). No `_LAZY` dicts, no
`__getattr__` fallbacks.

### Why `compile.embedding.dim` was removed

The `dim = 384` fallback shaped only the empty-text dedup vector while non-empty
fingerprints used `provider.embed()`'s real dimension, so a mismatch crashed
`EmbeddingMatcher.score(..., strict=True)` — and `make_provider()` ignored the
hint anyway. Both zvec stores had the same literal; all are gone and dim always
resolves from `provider.dim`, lazily probed. Stores rebuild when the stored
dimension differs from the current model's.

### Why `_ENV_LIST_VARS` is schema-derived

The hand-listed `KOMP_*` names drifted from the schema (splitting a `str` field
into a list, listing a dead var). It now derives from list-typed config-map
fields. Separately, `KOMP_CHARS_PER_TOKEN` was a dead env var — the real alias is
`KOMP_LLM_DEFAULT_CHARS_PER_TOKEN` and the value already flows via
`default_chars_per_token`.

### Lint remediation collapsed to page generation (2026-08-14)

Three overlapping tiers became one: LLM-generate missing concept pages from
cached extractions. Image-embed fixing moved into `run --fix` as the
`image-embed` structural rule; recommendations live only in `kompend lint gaps`;
`RemediationAction.tier` was removed. Same change added the `rumdl` structural
rule (`rumdl check` JSON over the vault's markdown, sources never touched, files
passed explicitly to bypass rumdl's ignore filtering).

### Release versioning

Calver `YYYY.M.D` / `YYYY.M.D.N` — no zero-padding, no `v` prefix. See
`.forgejo/workflows/release.yml`.

### Fixed CLI violations archive (moved from `src/kompend/cli/AGENTS.md`)

| Fixed | Was |
|---|---|
| `_build_router()` -> `build_default_router()` in `kompend.preprocessing.parsers._registry` | Parser construction in CLI |
| `_load_wiki()` -> `load_cached_wiki()` in `kompend.core.models` | JSON deserialization in CLI |
| `_build_vector_index()` -> `build_vector_index()` in `kompend.rag` | RAG business logic in CLI |
| `status.py` now uses `find_project_config()` | Hardcoded `.kompend/config.yaml` path |
| `SUPPORTED_EXTENSIONS` -> `get_supported_extensions()` querying parser registry | Duplicated extension set |
