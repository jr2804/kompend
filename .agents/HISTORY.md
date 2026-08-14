# AGENTS.md — History

Past design decisions, known issues, and "why we banned X." Optional background
reading; not required for day-to-day work.

## General History

### Pattern: Design Decision Log

Record significant decisions that shaped the codebase. Each entry explains the
context, the decision, and the commit where it landed. This prevents future
contributors from re-litigating settled questions.

### Rules

- **Append-only.** New entries at the top, never delete old ones.
- **Resolved entries.** Move to a "Resolved" subsection rather than removing.
- **Date + commit.** Every entry includes a date and the relevant commit hash.

### Extension

| Who | When | What |
|-----|------|------|
| Any contributor | Making a notable design decision | Add entry at top |
| Project maintainer | Resolving a known violation | Move to Resolved subsection |
| Project maintainer | Deprecating a tool/pattern | Add rationale entry |

---

## Project-specific History — KNOX

### Lint UX: `remediate` collapsed to LLM page generation (2026-08-14)

`knox lint remediate` previously had three tiers: image-embed auto-fix
(Tier 1), LLM coverage-gap generation (Tier 2), and recommendations
(Tier 3). The tiers overlapped confusingly with `run --fix` and `gaps`.
Remediation now does one thing: LLM-generate missing concept pages from
cached extractions. Image-embed fixing moved into `run --fix` as the new
`image-embed` structural rule (which also stopped `broken-wikilinks` from
flagging/stripping correct `![[img.png]]` embeds). Recommendations live only
in `knox lint gaps`. `RemediationAction.tier` field removed. Docs updated:
`docs/guides/lint.md`, `docs/guides/cli.md`.

Also added the `rumdl` structural rule: runs `rumdl check` (JSON output) on
all `*.md` files under the vault root except `sources/`, `.knox/`, and
`attachments/`; `--fix` applies `rumdl check --fix` (self-written notes get
reformatted, sources never touched). Files are passed explicitly to rumdl
to bypass its ignore-file filtering — dir scans get eaten by `.gitignore`.

### Plans completed and archived (2026-08-14)

`MIGRATION_PLAN.md` (summarization-pydantic-ai integration) and
`REFACTORING_PLAN.md` (super-module splits) moved to `.agents/history/`.

- MIGRATION_PLAN: fully implemented — `summarization-pydantic-ai>=0.1.10` in
  pyproject, `ContextManagerCapability` wired in `agents/base.py`,
  `_compact_messages` removed (commit `edb4d49`), `compile.llm.context_manager`
  config key added.
- REFACTORING_PLAN: Phases 1–4 + cleanup item 12 landed (pipeline `stages/`,
  config `engine/`, mcp `tools/`, compilation `flows/`, web routes,
  `_query_helpers`, `_sources_io`, `_post_compile`, `_checkpoints`,
  `_remediation_gaps`, output `pages/`). Remaining Phase-5 targets
  (`graph.py`, `wicer.py`, `dedup/orchestrator.py`) re-measured under the
  350-net-line budget (335/308/184) — splits no longer needed.

### Why `compile.embedding.dim` was removed

The config fallback `dim = 384` shaped only the empty-text vector in the dedup
embedding matcher; non-empty fingerprints came from `provider.embed()` at the
provider's real dimension, so a mismatch crashed `EmbeddingMatcher.score`
(`zip(..., strict=True)`). `dim` was also a no-op hint — `make_provider()`
ignores it. Fix: the matcher reads `provider.dim` (lazily probed at runtime)
and the config key was deleted. Commit: uncommitted (worktree after `fe517d1`).

Same principle applied to the zvec stores (`rag/zvec_store.py`,
`memory/store.py`): the `embedding_dim=384` defaults were the last
`dim = 384` literals (AGENTS.md antipattern table). Removed entirely — the
`embedding_dim` params/attributes and `make_provider(dim=)` are gone; dim
always resolves from `provider.dim`. Additionally, both stores now rebuild
instead of reopening when an existing collection's stored dimension differs
from the current model (previously raised "field dimension mismatch" on
insert, e.g. after switching from a 384-dim to a 768-dim embedding model).

### Why `_ENV_LIST_VARS` is schema-derived and `KNOX_CHARS_PER_TOKEN` was dropped

`_ENV_LIST_VARS` hand-listed KNOX_* names and drifted from the schema: it split
`KNOX_DEDUP_MATCHERS` (a `str` field) and `KNOX_SEARCH_EXPANSION_VARIANTS` (an
`int` field) into lists and listed dead `KNOX_LINT`. It now derives from
list-typed config-map fields. Separately, `_two_phase_core` read
`KNOX_CHARS_PER_TOKEN` by hand while the real config alias is
`KNOX_LLM_DEFAULT_CHARS_PER_TOKEN` — the direct read was redundant (the value
already flows via `default_chars_per_token`) and the env var was effectively
dead. Removed. Commit: uncommitted (worktree after `fe517d1`).

### Known text-in-code violations (T1–T10)

Tracked via `bd` (beads). Check current status: `bd list`.

These are LLM prompt strings that live inline in Python modules instead of in
Jinja2 templates under `src/knox/packs/`. They must be migrated before adding
new text-in-code.

| Issue | File | What |
|-------|------|------|
| `knox-ox2` (T1) | `lint/llm_rules.py` | 3 LLM prompt strings for lint rules |
| `knox-8b0` (T2) | `packs/base.py` | 4 default pack prompt strings |
| `knox-1a8` (T3) | `packs/builtins/academic.py` | 3 academic pack prompt strings |
| `knox-65z` (T4) | `packs/builtins/technical.py` | 3 technical pack prompt strings |
| `knox-174` (T5) | `compilation/schema.py` | BUILTIN_KINDS dict with descriptions |
| `knox-dnu` (T6) | `compilation/two_phase.py` | 4 extraction/generation prompt strings |
| `knox-844` (T7) | `compilation/wicer.py` | 3 WiCER loop prompt strings |
| `knox-lf5` (T8) | `verification.py` | 2 claim verification prompt strings |
| `knox-8gj` (T9) | `output/wiki_index.py` | 1 synthesis prompt string |
| `knox-sga` (T10) | `resolution/llm.py` | 1 default entity extraction prompt |

---

### Why text-in-code is banned

Historically, KNOX embedded LLM prompts directly in Python modules. This made
prompts hard to discover, version, and maintain across packs. The rule now is
that all LLM prompts must live in `packs/<name>/templates/` as Jinja2 files.
The T1–T10 violations above are the remaining legacy cases.

---

### Why litellm was removed

Commit `c8055c0`. KNOX originally used litellm for multi-provider LLM dispatch.

### Lazy-loader (SPEC-0001) for heavy imports

`knox --help` took ~80s because command modules eagerly imported heavy deps
(docling/torch/transformers ~16s, datasketch/scipy ~5s, pydantic-ai ~3s,
crawl4ai ~1s, MCP SDK ~1.5s). Root cause: `register_all_commands()` imports
all 28 CLI command modules at module scope, each pulling transitively heavy
optional deps through `__init__.py` files.

Fix: adopted `lazy-loader` (scientific-python SPEC-0001) — the standard
solution battle-tested in scikit-image, scipy, librosa, napari.
`lazy.attach(__name__, submod_attrs={...})` in `__init__.py` defers submodule
loading until first attribute access. Applied to 5 packages: `knox.acp`,
`knox.llm`, `knox.preprocessing.parsers`, `knox.preprocessing.dedup`,
`knox.preprocessing.downloaders`. For heavy classes used only inside function
bodies, the parser registry's `get_backend("name")(ParserBuildOptions())`
factory pattern is used instead of module-scope imports.

Result: `--help` 80s → 11s (−86%). Replaces hand-rolled `_LAZY` dicts +
`__getattr__` in `acp/__init__.py` and `llm/__init__.py`. `.pyi` stubs via
`lazy.attach_stub()` solve the static-checker type gap (deferred).

Rule added to `.agents/POLICIES.md`.
After migrating to PydanticAI's native `infer_model()` and the models.dev
catalog, litellm became dead weight — it pulled 40+ transitive dependencies
and added latency. Removed in favor of the unified `build_model()` path in
`src/knox/llm/`.

---

### Release versioning

Releases follow `YYYY.M.D` / `YYYY.M.D.N` calver format (no zero-padding, no
`v` prefix). See `.forgejo/workflows/release.yml`.
