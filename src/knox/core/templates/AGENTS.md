# DOX — knox.core.templates

<!-- on entry: loaded when agent enters src/knox/core/templates -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Layered Jinja2 template loading with frontmatter parsing and variable
validation. Loads templates from project → user → built-in locations.

## Modules

| Module | Contents |
|--------|----------|
| `loader.py` | `TemplateLoader`, `TemplateMeta`, `TemplateSource`, `get_template_loader`, `_parse_frontmatter` |
| `__init__.py` | Re-exports public API — import `from knox.core.templates import get_template_loader` |

## Rules

- Import from the package root: `from knox.core.templates import get_template_loader`.
- Template files use `.md.j2` extension with YAML frontmatter. See
  `src/knox/AGENTS.md`: Jinja2 conventions.
- The loader depends on `knox.core.config._core.find_project_config` to
  locate the project root. This is an acceptable coupling — template lookup
  is inherently project-root–relative.
