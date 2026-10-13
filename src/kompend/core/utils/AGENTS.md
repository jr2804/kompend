# DOX — kompend.core.utils

<!-- on entry: loaded when agent enters src/kompend/core/utils -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Pure utility helpers with zero domain logic. No imports from pipeline,
config, or models — only standard library and `kompend.logging`.

## Modules

| Module          | Contents                                                                              |
| --------------- | ------------------------------------------------------------------------------------- |
| `time.py`       | Local-timezone timestamp helpers: `now_local`, `iso_local`, `display_local`           |
| `exceptions.py` | `shorten_exception`, `shorten_string`, `MAX_CHARS_EXCEPTION`, `LLMOutputError`        |
| `dicts.py`      | Dotted-key dict helpers: `get_dot`, `set_dot`, `nest_flat`, `flat_keys`, `deep_merge` |
| `tokens.py`     | Char-based token estimate: `estimate_tokens`                                          |

## Rules

- **No domain imports.** These modules must not import from `kompend.core.models`,
  `kompend.core.pipeline`, or `kompend.core.config`. They are leaf utilities.
- Use `kompend.core.utils.time` / `kompend.core.utils.exceptions` at the import site.
