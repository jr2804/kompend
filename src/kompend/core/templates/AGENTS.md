# DOX — kompend.core.templates

<!-- on entry: loaded when agent enters src/kompend/core/templates -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Layered Jinja2 template loading with frontmatter parsing and variable
validation. Owns the prompt-template contract for the whole source tree: how a
template file is shaped, where the loader looks for it, and what happens when a
declared variable is missing.

## Ownership

- `src/kompend/core/templates/` — `TemplateLoader`, `TemplateMeta`, `TemplateVariable`
- The prompt-template format itself (extension, frontmatter schema, validation)

## Architecture

| Module | Contents |
| ------ | -------- |
| `loader.py` | `TemplateLoader`, `TemplateMeta`, `TemplateSource`, `get_template_loader`, `_parse_frontmatter` |
| `__init__.py` | Re-exports the public API |

Search order (first match wins):

| Priority | Location | Example |
| -------- | -------- | ------- |
| 1 (highest) | Project overrides next to `kompend.toml` | `./kompend/templates/` |
| 2 | User global overrides | `~/.config/kompend/templates/` |
| 3 (lowest) | Built-in package templates | `importlib.resources` in `kompend/templates/` |

```python
from kompend.core.templates import get_template_loader

loader = get_template_loader()
rendered = loader.render("lint/clarity-score.md.j2", title="My Page", content="...")
```

## Local Contracts

### Jinja2 conventions

- Template files use the `.md.j2` extension (markdown + Jinja2) and live in
  `src/kompend/templates/<module>/`.
- Variables use `{{ snake_case_name }}`; control flow uses `{% for %}` /
  `{% if %}` blocks, never Python string concatenation.
- Every template **starts with a YAML frontmatter block** delimited by `---`,
  parsed by `TemplateLoader` and absent from the rendered output:

```text
---
variables:
  - name: title
    required: true
  - name: content
    required: true
  - name: context_budget
    required: false
role: lint.factual-consistency
response_format: json_object
temperature: 0.0
---
```

- `variables` — expected variables with `name` and `required`.
- `role` — pipeline role (e.g. `"lint.clarity-score"`).
- `response_format` — `"json_object"` or `"text"`.
- `temperature` — per-template LLM temperature override.

### Validation

- **StrictUndefined** — any `{{ var }}` must have a matching kwarg; a missing
  variable raises `jinja2.UndefinedError` at render time.
- **Required variables** — a `required: true` variable missing from the kwargs
  raises `ValueError` *before* rendering.
- **Unused variable** — kwargs the template never references are logged at
  `WARNING` (catches typos like `contnt` vs `content`).
- Introspection uses `jinja2.meta.find_undeclared_variables()`, so validation
  never has to render the template.
- Callers may filter kwargs to declared variables; `KompendAgentBase` does
  (see `agents/AGENTS.md`). Add new variables to the template's `variables:`
  list.

### Rules

- Import from the package root: `from kompend.core.templates import get_template_loader`.
- No prompt text in Python — see `src/kompend/AGENTS.md` ("no text in code").
- The loader depends on `kompend.core.config._core.find_project_config` to locate
  the project root. Acceptable coupling — template lookup is inherently
  project-root-relative.

## Work Guidance

Before adding a string to a Python file, ask: is it human-readable text? An LLM
prompt? Configuration data? Then it belongs in a template (prompts) or in TOML
(config data), not in code. Short status and code-referencing error messages are
the only exceptions.
