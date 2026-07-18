# DOX — knox.core.utils

<!-- on entry: loaded when agent enters src/knox/core/utils -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Pure utility helpers with zero domain logic. No imports from pipeline,
config, or models — only standard library and `knox.logging`.

## Modules

| Module | Contents |
|--------|----------|
| `time.py` | Local-timezone timestamp helpers: `now_local`, `iso_local`, `display_local` |
| `exceptions.py` | `shorten_exception`, `shorten_string`, `MAX_CHARS_EXCEPTION`, `LLMOutputError` |

## Rules

- **No domain imports.** These modules must not import from `knox.core.models`,
  `knox.core.pipeline`, or `knox.core.config`. They are leaf utilities.
- Use `knox.core.utils.time` / `knox.core.utils.exceptions` at the import site.
