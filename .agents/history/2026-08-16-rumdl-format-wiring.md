# rumdl wiring: default config, compile-time format, `lint format` command

Created: 2026-08-15
Status: **implemented** (2026-08-15). All phases done; 1689 unit tests pass, ruff at baseline, mkdocs strict clean.

Driven by the p863 wiki lint run (2026-08-15T03:21): 30.2% health, 405/406
failures are rumdl style violations on **generated** output. Root causes
verified in code:

| Finding | Root cause (verified) |
|---|---|
| MD013 (160) | No rumdl config exists → rumdl default `line_length = 120`; generated reference lines run to 486 chars. |
| MD025 (58) | `front-matter-title` defaults to `^\s*title\s*[:=]` → every concept page's frontmatter `title:` counts as a second H1. Also `okf_writer._write_index` emits `# {type_name}` group headings (two H1s in bundle index). |
| MD066 (125) | `okf_writer._format_source_footnotes` emits `[^id]: …` definitions, but nothing rewrites the body's `[sources/x.pdf]` citations into `[^id]` references (docstring at `okf_writer.py:351` already promises this — never implemented). |
| MD005 (57) | `pages/index.py:generate_index` hardcodes `"  {i}. [[…]]"` (2-space indent); root index is written raw, never formatted. |
| index pages unformatted | `VaultWriter.write` (vault.py:258) writes concept pages but `_write_index`/`_write_log`/`_write_references_page` write raw markdown; `OkfWriter` never uses `DocumentRenderer` at all; `pages/index.py:write_index` (root landing) is raw. |

What already exists (no duplication to remove — gap is coverage, not code):

- `DocumentRenderer._format_via_rumdl` (renderer.py:99) pipes MarkdownWriter concept pages through `rumdl fmt --flavor … --stdin --config …`.
- `RumdlRule.fix` (structural.py:675) runs `rumdl check --fix` under `lint run --fix`.
- `compile.output.rumdl_config` config key + env alias already exist (`keys.py:129`, `_configs.py:237`); `knox.toml.example:452` documents `.knox/rumdl.toml` as the example.
- rumdl 0.2.55 installed; verified empirically: `front-matter-title = ""` (kebab-case, string regex) suppresses MD025 on frontmatter+body-H1 pages.

## Portability directive (binding, from user)

The wiki must be **self-contained and portable**: the knowledge lives in the
`wiki/` folder alone; it must work without `.knox/` (internal, gitignored
state) and without Obsidian (standard Markdown must render it; Obsidian must
render it too). Consequences:

1. **Link forms** — no Obsidian-only syntax as the primary form:
   - OKF bundle links `[Title](/concepts/x.md)` (leading slash, okf_writer.py
     `convert_wikilinks`) → **relative** `[Title](concepts/x.md)` — renders in
     Obsidian, GitHub, any md viewer. Flag: deviates from OKF §6.1
     "absolute links" wording; portability wins, note in docs.
   - The Obsidian-flavored `[[wikilinks]]` mode disappears with the Phase 0
     writer merge — relative links are the single link form.
   - Root index `generate_index` `[[slug|Title]]` (pages/index.py:52) →
     `[Title](wiki/concepts/<slug>.md)`.
2. **Index as catalog** (Karpathy gist, "Indexing and logging"): index.md is
   the query entry — a catalog of pages with link + one-line summary +
   metadata. The OKF bundle index already has `- [Title](path) - description`;
   the **root** index lacks summaries → add the frontmatter `description` to
   each TOC entry. Root index then links to the bundle index (`[Bundle](wiki/index.md)`)
   instead of duplicating the 57-link TOC.
3. **No `.knox/` dependency for query** — Phase 6.

## Phase 0 — OKF is a compliance spec, not an output format

User clarification (binding): OKF = compatibility with Google's Open
Knowledge Base format
(<https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md>).
It is **not** a format sibling of markdown/json/typst. `format = "okf"`
must be removed.

Current state (verified):

- `OutputFormat` enum (`_configs.py:105-116`): `OKF = "okf"` is the default
  member; `_configs.py:234` defaults `compile.output.format` to it.
- `OkfWriter(MarkdownWriter)` (`okf_writer.py:65`, `name = "okf"`) registers
  as a separate *format* in the writer registry (`base.py:__init_subclass__`);
  `_factory.py:44` looks up writers by this name.
- `src/knox/AGENTS.md:198-229` "OKF compliance — output format rules"
  enshrines the misconception (default format OKF; Obsidian MarkdownWriter
  "exempt" from rules 2–6).

Changes:

1. `OutputFormat` enum: remove `OKF` member; `MARKDOWN` becomes the default.
   Existing configs that explicitly set `format = "okf"` fail validation
   with a clear message listing valid formats (p863 wiki doesn't set it —
   defaults are unaffected; no deprecation shim per "delete over explain").
2. Merge `OkfWriter` into `MarkdownWriter`: the OKF-compliant bundle **is**
   the markdown output. `okf_writer.py` deleted; its behavior (OKF
   frontmatter schema, §8 index, §9 log, references page, relative links)
   folds into `markdown.py`. Registry shrinks to `markdown`, `json`, `typst`.
   The Obsidian-wikilink mode dies with it (portability directive already
   bans `[[…]]`); `flavor` remains a `rumdl fmt --flavor` knob only.
3. `src/knox/AGENTS.md:198-229`: rewrite — OKF is a compliance target of the
   default markdown output; drop the "exempt MarkdownWriter" carve-out.
4. `src/knox/output/AGENTS.md` architecture table: `OkfWriter` row →
   `MarkdownWriter (OKF-compliant bundle)`.
5. Docs: `docs/guides/output.md` (format section: okf no longer a choice),
   `docs/index.md`, `knox.toml.example:431-435`, `docs/concepts/open-knowledge-format.md`
   (reframe as compliance spec, not format).
6. Tests: `test_output_okf.py` stays but targets the merged writer
   (`BaseOutputWriter.get("okf")` assertion → `get("markdown")` + compliance
   checks); remove okf-format choices in `test_output*.py` fixtures.

Phases 2/3/6 below target the merged `MarkdownWriter` (not `okf_writer.py`).

## Phase 1 — shipped default config + `.knox/rumdl.toml` materialization

1. **`src/knox/lint/rumdl.toml`** (ships via hatchling package data like `src/knox/acp/builtin/*.toml`; no-text-in-code rule — config lives in a TOML file, not Python):

   ```toml
   # KNOX default rumdl config for generated wiki output.
   # Copy to .knox/rumdl.toml and edit to override.
   [MD013]
   line_length = 200
   [MD025]
   front-matter-title = ""   # frontmatter `title:` must not count as a second H1
   ```

   MD066 stays **enabled** — fixed properly in Phase 3 (footnotes get referenced). MD005 needs no config: `rumdl fmt`/`check --fix` auto-fix it (Phase 2 + existing `lint run --fix`).
2. **`src/knox/core/paths.py`**: add `RUMDL_CONFIG_FILE: Final[str] = f"{KNOX_DIR}/rumdl.toml"` (path constants live here per FILES.md).
3. **`_configs.py:237`**: `rumdl_config` default `None` → keep `None` (semantics: "use vault default"). New resolver `resolve_rumdl_config(base_dir, configured)` in `src/knox/lint/structural.py` (or `core/config/`):
   - `configured` set → resolve as-is (back-compat; `lint.py:136` passes `cfg.get(COMPILE_OUTPUT_RUMDL_CONFIG) or None`).
   - unset → `base_dir / RUMDL_CONFIG_FILE`; if missing, materialize by copying the shipped `rumdl.toml` (via `importlib.resources`, same pattern as `packs/base.py:152`) and log "wrote .knox/rumdl.toml (edit to override)".
4. **Use sites**: `_factory.py:127` (writer construction) and `lint.py:136` (lint runner) both switch to the resolver. Renderer/fmt and RumdlRule then pass the resolved path via existing `--config` plumbing.
5. `knox.toml.example:450-452`: update comment — default is now `.knox/rumdl.toml`, auto-materialized on first use.

## Phase 2 — `rumdl fmt` wired into compile (all generated pages)

Lazy-correct approach: one fmt pass over the whole generated tree after write, instead of threading renderers into 4 writers.

1. New helper `format_output(output_dir: Path, files: list[Path], rumdl_config: Path) -> list[Path]` (in `src/knox/output/renderer.py` or `_post_compile.py`): runs
   `rumdl fmt --flavor <flavor> --config <cfg> <output_dir> <root index.md>` (flavor auto-detected; `--config` explicit because rumdl discovers config from the *file's* dir, and `.knox/rumdl.toml` would never be found by discovery).
   - Skip silently when rumdl binary missing (same guard as `_check_rumdl`); return changed files.
2. Call it in `src/knox/cli/_post_compile.py` after `write_index(...)` (line 103), with `output_dir` + the root `index.md`. Covers: OKF concept pages, bundle `index.md`/`log.md`/`references.md`, root `index.md`, and MarkdownWriter output (idempotent on already-formatted pages).
3. **MD025 structural fix in bundle index**: `okf_writer.py:216` `f"# {type_name}\n"` → `f"## {type_name}\n"` (single H1 `# Knowledge Bundle`).
4. **MD005 structural fix in root index**: `pages/index.py:generate_index` `f"  {i}. …"` → `f"{i}. …"` (no indent). (fmt would also fix it, but the generator should emit clean markdown in the first place — same cost.)

## Phase 3 — MD066: emit `[^id]` references for inline citations

1. Shared helper (in `okf_writer.py`, used by both writers): `rewrite_citations_to_footnotes(body, sources)`:
   - Regex `\[sources/([^\]\s,]+?)(?:\.\w+)?\]` → `[^<stem>]` when `<stem>` (lowercased for match) equals a `sources[].id` that `_format_source_footnotes` will emit; multi-source `[sources/a.pdf, sources/b.pdf]` → `[^a][^b]`.
   - Leave unmatched citations as plain text (no orphan refs — a definition-less `[^x]` would be worse than a readable citation).
2. Call in `OkfWriter._render_document` (okf_writer.py:137) before appending `_format_source_footnotes`; same in `MarkdownWriter` if it emits source footnotes (verify — markdown.py uses `format_bibliography_entry`; likely only OKF needs it).
3. Result: every footnote definition referenced → MD066 passes under default config; OKF §5.1 per-claim attribution actually works (restores the docstring's promise).

## Phase 4 — `knox lint format` subcommand (advice: subcommand, not top-level)

**Recommend `knox lint format`**, sibling of `run`/`gaps`/`remediate`/`prune`:

- Mirrors rumdl's own `check`/`fmt` pair and `lint run` (check) ↔ `lint format` (fmt).
- Reuses the `lint_app` typer group, `vault_paths_from_ctx`, `resolve_config`, progress bar, and Phase 2 helper. No new top-level command plumbing, no doc/registration surface.
- Semantics: `rumdl fmt --config <resolved> <output_dir> <root index.md>`; flags: `--check` (exit 1 if changes needed — CI), `--dry-run` (diff only), `--config` override.
- Alternative `knox format` top-level is cleaner naming but adds surface for zero functional gain; a 10-line wrapper can be added later if desired.

## Phase 6 — portable query (index-first, no `.knox/` dependency)

Current: `query_cmd` (knowledge/query.py:106) requires
`load_cached_wiki(vp.base_dir)` (`.knox/` cache) + `open_vector_store`
(`.knox/vectors/`); without `.knox/` it exits with "No compiled wiki found"
— the knowledge sits in `wiki/` but is unreachable. Violates the gist's
index-first design and the portability directive.

1. **`load_wiki_from_files(output_dir) -> CompiledWiki`** (new, in
   `src/knox/core/models/wiki.py`): builds `CompiledWiki` from the generated
   markdown alone — parse frontmatter (title/description/type via existing
   yaml path), body, and links (`[x](path)` and `[[x]]`) into
   `WikiNode`/`GraphEdge`. No LLM, no cache.
2. **Index-first retrieval** (gist): parse the bundle `index.md` catalog
   (title + summary + link) and rank candidates by title/summary match
   against the query; drill into matched pages; content keyword match as
   fallback. This is the primary portable path — embedding RAG stays as the
   `.knox/`-backed accelerator, never a requirement.
3. `query_cmd`: `wiki = load_cached_wiki(...) or load_wiki_from_files(vp.output_dir)`;
   only error when neither exists. `open_vector_store` only when cache present;
   portable path skips it (FTS/keyword over loaded pages).
4. Tests: tmp vault with **only** generated md files (no `.knox/`) → query
   returns ranked pages from index catalog + content.
5. `log.md` conformance (gist: append-only `## [date]` entries, grep-able):
   current `_write_log` is a one-shot creation record. Small change: emit
   `## [<date>] <action>` prefixed entries; optionally append query/lint
   activity. Scope flag: do this only if the append-only activity log is
   wanted now, else leave as follow-up.

## Phase 5 — verification

- `tests/unit/test_lint.py` (extend RumdlRule section, line ~531): resolver returns `.knox/rumdl.toml` by default; materializes missing file from shipped default; explicit config wins.
- `tests/unit/test_output_okf.py`: citation rewrite (single, multi-source, unmatched, no-sources); bundle index has single H1.
- `tests/unit/test_output_wiki_index.py`: root index emits `1. …` unindented; root index linked correctly; root index TOC entries carry summaries.
- `tests/unit/test_output_okf.py` + `test_output.py`: links are relative (`concepts/x.md`, no leading slash, no `[[…]]`) in both okf and markdown output.
- Portable query: tmp vault with only generated md files → `query_cmd` path returns ranked pages (index catalog match + content fallback); `.knox/` present → unchanged hybrid path.
- Format helper: tmp vault fixture → `rumdl check --config` on compiled output → zero MD013/MD025/MD066/MD005 findings.
- `ruff check`, `uv run pytest tests/unit -x --timeout=30`, `uv run pytest tests/integration -x --timeout=120` if a compile-path test exists.
- DOX: `src/knox/output/AGENTS.md` (fmt coverage now includes index/log/overview/references + default config), `knox.toml.example`, `.agents/FILES.md` (add `RUMDL_CONFIG_FILE`).

## Follow-up (out of scope, flagged)

- Lint-run hangs on `ollama/gemma4:e2b` clarity-score timeouts (790 s run) — environmental; `--no-llm` avoids it.
