# DOX — kompend.core

<!-- on entry: loaded when agent enters src/kompend/core -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Core data models, pipeline orchestration, configuration, caching, and shared
utilities for the kompend pipeline. This is the foundational layer that all
other kompend subsystems build on.

## Ownership

- `src/kompend/core/` — shared foundation (models, paths, config, pipeline, utils)
- No business logic belongs here beyond pipeline orchestration and data model
  serialization

## Local Contracts

- Import paths follow the nested-package layout; import from the package root
  (e.g. `from kompend.core.models import Chunk`), not from individual submodules
- `core/text/` holds pure text utilities with no `kompend.*` imports at module
  scope (`matching.py` similarity, `encoding.py` UTF-8 detection,
  `rumdl.py` markdown normalisation). They are the only sanctioned place for
  string-in/string-out helpers shared across tiers.
- One normaliser per concern: markdown formatting goes through
  `core.text.rumdl` (text/batch variants) or `output.renderer.format_output`
  (file sweep) — never a second hand-rolled formatter
- Path constants — see `.agents/FILES.md`; new-code checklist — see
  `.agents/CHECKS.md`
- Python antipatterns — see `.agents/CODE.md`; text-in-code — see
  `src/kompend/AGENTS.md`

## Child DOX Index

| Path | Scope |
|------|-------|
| `page_builder.py` | Wenmode-based Markdown AST builder (shared by core and output modules) |
| `config/AGENTS.md` | Configuration system — resolution, precedence, TOML schema |
| `models/AGENTS.md` | Data models — enums, elements, chunks, structured data, trees, wiki, citations |
| `pipeline/AGENTS.md` | Pipeline orchestration — orchestrator, config, state, ingest |
| `resolution/AGENTS.md` | Programmatic Tier 3 resolution — entity registry, cross-reference checker |
| `cache/AGENTS.md` | Content-addressed cache store |
| `store/AGENTS.md` | Single non-vector SQLite store (`.kompend/kompend.db`) — schema, connection, clean-cut policy |
| `templates/AGENTS.md` | Layered Jinja2 template loading |
| `trust/AGENTS.md` | Actor identity and trust/privacy |
| `reporting/AGENTS.md` | CLI observability reports — log, runlog |
| `utils/AGENTS.md` | Pure utilities — time, exception formatting |
