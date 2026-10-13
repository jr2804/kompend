# DOX — kompend.output

<!-- on entry: loaded when agent enters src/kompend/output -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Tier 5 output — render a ``CompiledWiki`` into multiple formats (Markdown,
JSON, Typst). The markdown writer is OKF v0.2-compliant by default. Also
builds the wiki index page and serves wiki page previews.

## Ownership

- ``src/kompend/output/`` — all output writers, renderers, and wiki index

## Architecture

| Module | Contents |
|--------|----------|
| ``base.py`` | ``BaseOutputWriter`` ABC with plugin registry (``register`` / ``get``) |
| ``renderer.py`` | ``DocumentRenderer`` — canonical Markdown rendering (wenmode AST, deterministic output); ``format_output`` — explicit ``kompend lint format`` helper |
| ``markdown.py`` | ``MarkdownWriter`` — the single markdown writer (OKF v0.2-compliant bundle, portable relative links, §5.1 footnote attribution) |
| ``vault.py`` | ``VaultWriter`` — shared vault algorithm: page routing, stale-clearing, citation rewriting, references/index/log orchestration, per-source summaries (``wiki/summaries/``), copy-on-embed image portability |
| ``json_writer.py`` | ``JsonWriter`` — JSON output |
| ``typst.py`` | ``TypstWriter`` — Typst output |
| ``wiki_index.py`` | Wiki index page construction |
| ``models.py`` | Output-specific data models |
| ``pages/`` | Wiki page sub-renderers |
| ``pages/index.py`` | Vault-root index page rendering (portable catalog) |
| ``pages/overview.py`` | Overview page rendering |
| ``pages/log.py`` | Log page rendering (``LogStage`` enum, ``generate_log_entry`` via ``PageBuilder``) |

## Local Contracts

### OKF v0.2 conformance

KOMPEND emits OKF v0.2 bundles; OKF is a **compliance spec**, not an output
format, and `compile.output.format` only offers real formats (`markdown`,
`json`, `typst`). Spec: <https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md>.
User-facing guide: `docs/concepts/open-knowledge-format.md`.

1. **Frontmatter** is built once by `VaultWriter.okf_frontmatter()` — every
   writer delegates to it (single source of truth). Every concept file has
   parseable YAML with a non-empty `type` (§11.2).
2. **Trust fields** — `generated: {by, at}` (§5.2; `by` follows the §7 actor
   convention `kompend/<version>`) supersedes the v0.1 `timestamp`. `verified` is
   emitted only when a verification pass ran; `status` defaults to `stable`.
3. **Provenance fields are distinct** — `tags` is topic tags only (no filename
   overload), `source_files` is the KOMPEND extension (provenance stems), `sources`
   is OKF §5.1 structured objects `{id, resource, title, author}` from the
   reference store, never bare strings.
4. **Reserved filenames** — `index.md` and `log.md` follow §8/§9. `log.md` is
   date-grouped `## YYYY-MM-DD` headings, newest first, entries as
   `- [HH:mm:ss] **Action**: prose`; it is never wiped — compile, ingest, query,
   and lint all append (`LogStage`). Root `index.md` may declare
   `okf_version: "0.2"`.
5. **Consumers MUST tolerate** missing optional fields, unknown `type` values,
   unknown frontmatter keys, broken cross-links, and a missing `index.md`
   (§10, permissive consumption).
6. **Page kinds** — mandatory taxonomy plus configured extras; see
   `compilation/AGENTS.md` (page kinds).

## Rules

- Import from the package root: ``from kompend.output import MarkdownWriter``.
- ``DocumentRenderer`` is the single path for per-page Markdown rendering;
  all generated output is additionally piped through ``rumdl fmt``
  (``format_output``) after write — no hand-rolled flavor converters.
- OKF is a **compliance spec**, not an output format: there is exactly one
  markdown writer, and it is OKF v0.2-compliant. Never add a second
  markdown variant for flavor/link-syntax reasons.
- Links are portable, folder-relative markdown paths — resolved
  ``[[wikilinks]]`` become ``[title](path.md)`` (alias-aware: acronym,
  bare-prefix, and substring matches count as resolved). Unresolvable
  wikilinks stay as ``[[target]]`` so broken cross-links remain visible.
- Footnote definitions are emitted only for ids actually referenced in the
  body, ordered by first reference (MD066/MD067-clean).
- Importing this package auto-registers all built-in writers via
  ``BaseOutputWriter`` subclass registration.
- ``VaultWriter._load_verification`` reads the latest ``verification`` row from
  ``.kompend/kompend.db`` (``ORDER BY id DESC LIMIT 1``), not ``.kompend/verification.json``;
  a missing pass yields ``None`` and the quality page is skipped.
