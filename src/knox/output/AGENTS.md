# DOX — knox.output

<!-- on entry: loaded when agent enters src/knox/output -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Tier 5 output — render a ``CompiledWiki`` into multiple formats (Markdown,
OKF bundles, JSON, Typst). Also builds the wiki index page and serves
wiki page previews.

## Ownership

- ``src/knox/output/`` — all output writers, renderers, and wiki index

## Architecture

| Module | Contents |
|--------|----------|
| ``base.py`` | ``BaseOutputWriter`` ABC with plugin registry (``register`` / ``get``) |
| ``renderer.py`` | ``DocumentRenderer`` — canonical Markdown rendering via rumdl |
| ``markdown.py`` | ``MarkdownWriter`` — Markdown output (obsidian, gfm flavors) |
| ``okf_writer.py`` | ``OkfWriter`` — OKF v0.1 compliant bundle output |
| ``json_writer.py`` | ``JsonWriter`` — JSON output |
| ``typst.py`` | ``TypstWriter`` — Typst output |
| ``wiki_index.py`` | Wiki index page construction |
| ``models.py`` | Output-specific data models |
| ``pages/`` | Wiki page sub-renderers |
| ``pages/index.py`` | Index page rendering |
| ``pages/overview.py`` | Overview page rendering |
| ``pages/log.py`` | Log page rendering |

## Rules

- Import from the package root: ``from knox.output import MarkdownWriter``.
- ``DocumentRenderer`` is the single path for all Markdown output.
- All flavor-specific Markdown formatting goes through ``rumdl fmt`` —
  no hand-rolled flavor converters.
- Importing this package auto-registers all built-in writers via
  ``BaseOutputWriter`` subclass registration.
