# DOX — knox.output

<!-- on entry: loaded when agent enters src/knox/output -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Tier 5 output — render a ``CompiledWiki`` into multiple formats (Markdown,
JSON, Typst). The markdown writer is OKF v0.2-compliant by default. Also
builds the wiki index page and serves wiki page previews.

## Ownership

- ``src/knox/output/`` — all output writers, renderers, and wiki index

## Architecture

| Module | Contents |
|--------|----------|
| ``base.py`` | ``BaseOutputWriter`` ABC with plugin registry (``register`` / ``get``) |
| ``renderer.py`` | ``DocumentRenderer`` — canonical Markdown rendering (wenmode AST, deterministic output); ``format_output`` — explicit ``knox lint format`` helper |
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

## Rules

- Import from the package root: ``from knox.output import MarkdownWriter``.
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
