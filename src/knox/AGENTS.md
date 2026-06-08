# AGENTS.md — knox/src (Python codebase)

Coding conventions for the KNOX Python source tree.

## No text in code

**LLM prompts, system instructions, template strings, configuration data, and
any other human-readable text must NOT be embedded in Python source files.**

This rule exists because:

- **Discoverability** — Text hidden in Python string constants is invisible to
  non-developers (domain experts, technical writers) who need to review prompts.
- **Editability** — Changing a prompt requires editing a Python file, which
  risks introducing syntax errors, import issues, or accidental code changes.
- **Versioning** — Prompts and code have different review cycles. A prompt tweak
  should not recompile the Python package.
- **Tooling** — Jinja2 templates can be syntax-checked, diffed, and previewed.
  TOML config can be validated with a schema. Python string constants cannot.

### What goes where

| Content | Format | Location |
|---------|--------|----------|
| LLM system prompts | Jinja2 (`.md.j2`) | `<module>/templates/<role>_system.md.j2` |
| LLM user prompts / templates | Jinja2 (`.md.j2`) | `<module>/templates/<role>_prompt.md.j2` |
| Page kind definitions (N9) | TOML | `<module>/kinds.toml` or config |
| Pack prompts and roles | TOML | `packs/<name>/pack.toml` |
| Any other human-readable string | TOML / Jinja2 | appropriate config dir |

### How templates are loaded

```python
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

_loader = FileSystemLoader(Path(__file__).parent / "templates")
_env = Environment(loader=_loader)

def _render(role: str, **kwargs: object) -> str:
    template = _env.get_template(f"{role}.md.j2")
    return template.render(**kwargs)
```

### Acceptable exceptions

- Short (≤5 words) status messages passed to `typer.echo()` or logger calls
- Error messages that must reference code-level constructs (variable names, types)
- Single words used as enum members, dict keys, or field names
- Docstrings and type annotations

## Known violations (T1–T10)

These are filed as beads issues and must be fixed before adding new text-in-code:

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

## Jinja2 conventions

- Template files use the `.md.j2` extension (markdown + Jinja2).
- Variables use `{{ snake_case_name }}` syntax.
- Control flow uses `{% for %}` / `{% if %}` blocks, never Python string concatenation.
- Each template file starts with a YAML frontmatter block (in `{# #}` Jinja2 comment)
  documenting the expected variables and response format:

```jinja2
{# ---
variables:
  - title: str        # page title
  - content: str      # page content up to context_budget
  - context_budget: int  # max chars to include
response_format: json_object
role: lint.factual-consistency
temperature: 0.0
--- #}
```

- Template directories contain an `_index.json` or `index.toml` listing all
  available templates with their metadata (role, expected variables, format).

## Adding new text — checklist

Before adding a string constant to a Python file, ask:

1. **Is this human-readable text?** If yes, it belongs in a template or config file.
2. **Is this an LLM prompt?** → Jinja2 template in `<module>/templates/`.
3. **Is this configuration data?** → TOML in `<module>/` or the global config.
4. **Is this a short status message?** → Acceptable in Python, but keep it ≤5 words.
5. **Is this an error message referencing code internals?** → Acceptable in Python.
6. **Have I added a `templates/` directory if one doesn't exist?** → Do it.

## Pack structure

When moving prompts out of `packs/builtins/<name>.py`, use this layout:

```
src/knox/packs/<name>/
├── pack.toml           # manifest + prompts (see TOML schema)
├── __init__.py         # optional Python registration
└── templates/          # optional Jinja2 templates
    └── compile.md.j2
```

The `pack.toml` manifest:

```toml
[pack]
name = "academic"
description = "Prompts for academic research papers"
version = "1.0.0"

[pack.dependencies]
# other pack names

[roles.compile]
template = "compile.md.j2"
response_format = "json_object"
temperature = 0.0

[roles.resolve]
template = "resolve.md.j2"
response_format = "json_object"
temperature = 0.0
```
