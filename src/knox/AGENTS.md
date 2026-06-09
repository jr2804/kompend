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

All templates go through ``knox.core.template_loader.TemplateLoader``,
which searches three locations in order (project > user > built-in):

| Priority | Location | Example path |
|----------|----------|-------------|
| 1 (highest) | Project overrides | ``./knox/templates/`` (next to ``knox.toml``) |
| 2 | User global overrides | ``~/.config/knox/templates/`` |
| 3 (lowest) | Built-in (package) | ``importlib.resources`` in ``knox/templates/`` |

Usage::

    from knox.core.template_loader import get_template_loader

    loader = get_template_loader()
    rendered = loader.render("lint/clarity-score.md.j2",
                              title="My Page", content="...")

### Template variable validation

Every template is validated at render time:

- **StrictUndefined** — Any ``{{ var }}`` referenced in the template *must* have
  a matching kwarg. Missing variables raise ``jinja2.UndefinedError`` immediately.
- **Required variable check** — If the template declares ``required: true`` in its
  frontmatter, the loader raises ``ValueError`` before rendering if the variable
  is missing.
- **Unused variable warning** — Kwargs that are never referenced in the template
  are logged at ``WARNING`` level. This catches typos like passing ``contnt``
  when the template expects ``content``.
- **AST introspection** — Uses ``jinja2.meta.find_undeclared_variables()`` to
  determine which variables the template actually uses, without rendering it.

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

## Python Antipatterns

1. **Follow the `python-ultimate` skill.** When in doubt about Python coding style, structure, or conventions, load and follow the guidelines in the `python-ultimate` skill. It takes precedence over any conflicting habit or default.

2. **No inline imports — PLC0415 is a hard error.** All imports must live at module level (standard library → third-party → local, per PEP 8 grouping). Importing inside functions, methods, or class bodies is forbidden. Using `# noqa: PLC0415` to suppress the lint error is itself an antipattern — it hides a real problem rather than fixing it. Move the import to the top of the file. The only accepted exceptions are cases where the import would cause a circular dependency or an unconditional heavy side-effect at import time; both must be documented with a comment explaining why.

3. **No try/except ImportError guards for core dependencies.** Every package listed in `[project.dependencies]` in `pyproject.toml` is guaranteed to be installed. Wrapping `import mylib` in `try: ... except ImportError: mylib = None` is pointless boilerplate — it obscures the real dependency, adds dead branches, and masks actual install failures. Import at module level, unconditionally.

## Jinja2 conventions

- Template files use the ``.md.j2`` extension (markdown + Jinja2).
- Variables use ``{{ snake_case_name }}`` syntax.
- Control flow uses ``{% for %}`` / ``{% if %}`` blocks, never Python string concatenation.
- Each template file **starts with a YAML frontmatter block** delimited by ``---``.
  This frontmatter is parsed by ``TemplateLoader`` and does not appear in the
  rendered output::

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

## temperature: 0.0

    ---

    Your prompt text using {{ title }} and {{ content }}.

  The frontmatter fields are:

- ``variables`` — list of expected variables with ``name`` and ``required``.
- ``role`` — pipeline role (e.g. ``"lint.clarity-score"``).
- ``response_format`` — LLM response format (``"json_object"`` or ``"text"``).
- ``temperature`` — LLM temperature override for this specific template.
- Required variables declared in frontmatter are checked by the loader. If a
  required variable is missing, ``ValueError`` is raised before the template
  is ever rendered.

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

    src/knox/packs/<name>/
    ├── pack.toml           # manifest + prompts (see TOML schema)
    ├── __init__.py         # optional Python registration
    └── templates/          # optional Jinja2 templates
        └── compile.md.j2

The `pack.toml` manifest:

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
