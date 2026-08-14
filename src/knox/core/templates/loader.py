"""Layered Jinja2 template loader for KNOX.

Loads templates from three sources in precedence order:

1. **Project templates** — ``<project_root>/knox/templates/`` (alongside ``knox.toml``)
2. **User global templates** — ``~/.config/knox/templates/`` (XDG_CONFIG_HOME)
3. **Built-in templates** — inside the installed ``knox`` package (``importlib.resources``)

Template variable validation
----------------------------

Every template is checked at render time:

- **StrictUndefined** — any ``{{ var }}`` referenced in the template *must* have a
  matching kwarg. Missing variables raise ``jinja2.UndefinedError`` immediately.
- **Unused variable warning** — kwargs that are provided but never referenced in
  the template are logged at ``WARNING`` level. This catches typos like passing
  ``contnt`` when the template expects ``content``.
- **AST introspection** — ``jinja2.meta.find_undeclared_variables()`` is used to
  determine which variables the template actually uses, without rendering it.
"""

from __future__ import annotations

import importlib.resources as ir
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jinja2 import BaseLoader, ChoiceLoader, Environment, FileSystemLoader, PackageLoader, StrictUndefined, meta

from knox.core.config._core import find_project_config
from knox.logging import get_logger
from knox.types import Jinja2Env, Jinja2Loader

# ── Built-in template root ──────────────────────────────────────────────

_BUILTIN_TEMPLATE_PACKAGE: str = "knox"
_BUILTIN_TEMPLATE_DIR: str = "templates"

# ── Search path constants ───────────────────────────────────────────────

_USER_CONFIG_DIR: str = "knox"
_USER_TEMPLATES_NAME: str = "templates"
_PROJECT_TEMPLATES_DIR: str = "knox"
_PROJECT_TEMPLATES_NAME: str = "templates"


# ── Frontmatter parsing ─────────────────────────────────────────────────

_FRONTMATTER_RE: re.Pattern[str] | None = None

logger = get_logger(__name__)

# ── Module-level singleton ───────────────────────────────────────────

_loader_instance: list[TemplateLoader] = []  # mutable container avoids global statement


# ── Dataclasses ─────────────────────────────────────────────────────────


@dataclass
class TemplateSource:
    """Identifies where a loaded template came from."""

    source: str  # "builtin", "user", "project"
    template_file: Path | str = ""


@dataclass
class TemplateMeta:
    """Parsed metadata for a template, extracted from its YAML frontmatter.

    Frontmatter is a YAML block delimited by ``---`` at the very top
    of the template file::

        ---
        variables:
          - name: title
            required: true
          - name: content
            required: true
        role: lint.factual-consistency
        response_format: json_object
        temperature: 0.5
        ---

        Your prompt text with {{ title }} and {{ content }}.
    """

    variables: list[dict[str, Any]] = field(default_factory=list)
    """Declared variables with name, type hint, and required flag."""

    role: str = ""
    """Pipeline role this template is for (e.g. ``\"lint.clarity-score\"``)."""

    response_format: str = ""
    """Expected LLM response format (``\"json_object\"`` or ``\"text\"``)."""

    temperature: float | None = None
    """LLM temperature override for this template."""
    raw: dict[str, Any] = field(default_factory=dict)

    def required_variable_names(self) -> set[str]:
        """Return the set of variable names marked as required."""
        return {v["name"] for v in self.variables if isinstance(v, dict) and v.get("required")}

    def declared_variable_names(self) -> set[str]:
        """Return the set of all declared variable names."""
        return {v["name"] for v in self.variables if isinstance(v, dict) and v.get("name")}

    @classmethod
    def from_yaml_block(cls, yaml_text: str) -> TemplateMeta:
        """Parse a YAML frontmatter string into a TemplateMeta."""
        try:
            data: dict[str, Any] = yaml.safe_load(yaml_text) or {}
        except Exception:
            return cls()

        return cls(
            variables=data.get("variables", []),
            role=data.get("role", ""),
            response_format=data.get("response_format", ""),
            temperature=data.get("temperature"),
            raw=data,
        )


# ── TemplateLoader ──────────────────────────────────────────────────────


@dataclass
class TemplateLoader:
    """Layered Jinja2 template loader with variable validation.

    Usage::

        from knox.core.templates import get_template_loader

        loader = get_template_loader()
        html = loader.render("lint/factual-consistency.md.j2", title="My Page", content="...")
        raw = loader.get_text("lint/factual-consistency.md.j2")
        meta = loader.get_meta("lint/factual-consistency.md.j2")

    Parameters
    ----------
    project_dir:
        Override the project root for template lookup. When ``None``
        (default), use :func:`_find_project_root` which walks up from
        ``cwd`` looking for ``knox.toml``.
    """

    project_dir: Path | None = None

    # Lazily built
    _jinja_env: Environment | None = None
    _search_cache: dict[str, tuple[str, Path | str]] = field(default_factory=dict)
    _meta_cache: dict[str, TemplateMeta] = field(default_factory=dict)
    # Variables injected into every render call (only if the template uses them)
    _global_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.project_dir is None:
            self.project_dir = _find_project_root()

    # ── Public API ───────────────────────────────────────────────────

    def set_global_context(self, **kwargs: Any) -> None:
        """Set variables that are injected into every ``render()`` call.

        Only variables that a template actually references are injected,
        so adding ``user_notes`` here will not trigger unused-variable
        warnings for templates that do not use it.

        Call :meth:`clear_global_context` when the context is no longer
        needed (e.g. after the pipeline finishes) to avoid leaking
        state across runs.
        """
        self._global_context.update(kwargs)

    def clear_global_context(self) -> None:
        """Remove all global context variables."""
        self._global_context.clear()

    def render(self, template_file: str, **kwargs: Any) -> str:
        """Load and render a Jinja2 template with variable validation.

        Parameters
        ----------
        template_file:
            Relative path inside the templates directory, e.g.
            ``"lint/factual-consistency.md.j2"``.
        **kwargs:
            Variables for the Jinja2 template.

        Returns:
        -------
        str
            Rendered output.

        Raises:
        ------
        jinja2.UndefinedError
            A variable referenced in the template was not provided.
        FileNotFoundError
            The template does not exist in any search path.
        """
        meta = self.get_meta(template_file)
        body_text = self._get_body(template_file)

        # Determine which variables the template actually uses
        used = self._find_undeclared_variables(template_file)

        # Inject global context for variables the template uses but the
        # caller did not provide.  This lets pipeline-wide variables like
        # ``user_notes`` flow into templates without threading them through
        # every call site.
        for key in used & self._global_context.keys():
            kwargs.setdefault(key, self._global_context[key])

        required = meta.required_variable_names()
        missing = required - kwargs.keys()
        if missing:
            raise ValueError(f"Template '{template_file}' requires missing variables: {', '.join(sorted(missing))}")

        extra = kwargs.keys() - used - set()
        if extra:
            logger.warning(
                "Template '%s' unused variables: %s. Expected: %s",
                template_file,
                ", ".join(sorted(extra)),
                ", ".join(sorted(used)),
            )

        # Render with StrictUndefined (catches any missing vars not in frontmatter)
        env = Environment(
            loader=self._make_template_source_loader(),
            autoescape=False,  # noqa: S701 — LLM prompts, not HTML
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )
        template = env.from_string(body_text)
        return template.render(**kwargs)

    def render_from_env(self, template_file: str, **kwargs: Any) -> str:
        """Render using the cached Jinja2 environment (no validation).

        Faster than :meth:`render` for hot paths, but skips frontmatter
        parsing and variable validation. Use when you have already
        validated variables elsewhere.
        """
        env = self._get_env()
        template = env.get_template(template_file)
        return template.render(**kwargs)

    def get_text(self, template_file: str) -> str:
        """Load raw template text (including any YAML frontmatter)."""
        source, _ = self._locate(template_file)
        return source

    def get_body(self, template_file: str) -> str:
        """Load template body text (with frontmatter stripped)."""
        return self._get_body(template_file)

    def get_meta(self, template_file: str) -> TemplateMeta:
        """Parse and return the template's YAML frontmatter metadata.

        Results are cached after first parse.
        """
        if template_file in self._meta_cache:
            return self._meta_cache[template_file]

        text = self.get_text(template_file)
        fm, _ = parse_frontmatter(text)
        meta = TemplateMeta.from_yaml_block("\n".join(f"{k}: {v}" for k, v in fm.items()))
        self._meta_cache[template_file] = meta
        return meta

    def locate(self, template_file: str) -> TemplateSource:
        """Return where a template was found (source label + path)."""
        _, identifier = self._locate(template_file)
        return identifier

    def list_templates(self) -> list[str]:
        """Return all available template paths (flattened across all layers)."""
        seen: set[str] = set()
        for _layer, paths in self._list_layers().items():
            for p in paths:
                if p not in seen:
                    seen.add(p)
        return sorted(seen)

    # ── Internals ───────────────────────────────────────────────────

    def _get_body(self, template_file: str) -> str:
        """Return template body with frontmatter stripped."""
        text = self.get_text(template_file)
        _, body = parse_frontmatter(text)
        return body

    def _find_undeclared_variables(self, template_file: str) -> set[str]:
        """Parse the template AST and return all referenced variable names.

        Uses ``jinja2.meta.find_undeclared_variables()`` which parses
        the template without rendering it.
        """
        body = self._get_body(template_file)
        env = Environment()  # noqa: S701 — only used for AST parsing, not rendering
        ast = env.parse(body)
        return meta.find_undeclared_variables(ast)

    def _locate(self, template_file: str) -> tuple[str, TemplateSource]:
        """Search project → user → builtin, return (source_text, source_info)."""
        normalized = template_file.replace("\\", "/")

        cached = self._search_cache.get(normalized)
        if cached is not None:
            label, identifier = cached
            if label in ("project", "user"):
                candidate_file = Path(identifier)
                if candidate_file.is_file():
                    return (candidate_file.read_text(encoding="utf-8"), TemplateSource(label, candidate_file))
                # Stale cache entry — clear and re-search
                del self._search_cache[normalized]

        # 1. Project templates
        project_tpl_dir = _project_templates_dir(self.project_dir)
        if project_tpl_dir is not None:
            candidate_file = project_tpl_dir / normalized
            if candidate_file.is_file():
                text = candidate_file.read_text(encoding="utf-8")
                self._search_cache[normalized] = ("project", str(candidate_file))
                return (text, TemplateSource("project", candidate_file))

        # 2. User global templates
        user_tpl_dir = _user_templates_dir()
        if user_tpl_dir is not None:
            candidate_file = user_tpl_dir / normalized
            if candidate_file.is_file():
                text = candidate_file.read_text(encoding="utf-8")
                self._search_cache[normalized] = ("user", str(candidate_file))
                return (text, TemplateSource("user", candidate_file))

        # 3. Built-in templates
        if _builtin_exists(normalized):
            text = _read_builtin(normalized)
            self._search_cache[normalized] = ("builtin", normalized)
            return (text, TemplateSource("builtin", normalized))

        self._search_cache[normalized] = ("not_found", "")
        raise FileNotFoundError(f"Template not found: {normalized} (searched project={project_tpl_dir}, user={user_tpl_dir}, builtin={_BUILTIN_TEMPLATE_DIR})")

    def _get_env(self) -> Jinja2Env:
        """Build and cache the Jinja2 environment (no frontmatter stripping).

        This environment uses ``StrictUndefined`` and expects templates
        stored *without* frontmatter in the search paths. Built-in templates
        must have frontmatter pre-stripped when stored.
        """
        if self._jinja_env is not None:
            return self._jinja_env

        loaders: list[BaseLoader] = []

        project_dir = _project_templates_dir(self.project_dir)
        if project_dir is not None:
            loaders.append(FileSystemLoader(str(project_dir)))

        user_dir = _user_templates_dir()
        if user_dir is not None:
            loaders.append(FileSystemLoader(str(user_dir)))

        try:
            loaders.append(PackageLoader(_BUILTIN_TEMPLATE_PACKAGE, _BUILTIN_TEMPLATE_DIR))
        except (ValueError, ModuleNotFoundError) as exc:
            logger.debug("No built-in templates directory: %s", exc, exc_info=True)

        self._jinja_env = Environment(
            loader=ChoiceLoader(loaders),
            autoescape=False,  # noqa: S701 — LLM prompts, not HTML
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )
        return self._jinja_env

    def _make_template_source_loader(self) -> Jinja2Loader:
        loaders: list[BaseLoader] = []

        project_dir = _project_templates_dir(self.project_dir)
        if project_dir is not None:
            loaders.append(FileSystemLoader(str(project_dir)))

        user_dir = _user_templates_dir()
        if user_dir is not None:
            loaders.append(FileSystemLoader(str(user_dir)))

        try:
            loaders.append(PackageLoader(_BUILTIN_TEMPLATE_PACKAGE, _BUILTIN_TEMPLATE_DIR))
        except (ValueError, ModuleNotFoundError) as exc:
            logger.debug("No built-in templates directory: %s", exc, exc_info=True)

        return ChoiceLoader(loaders)

    def _list_layers(self) -> dict[str, list[str]]:
        """Return all templates per layer for debugging."""
        layers: dict[str, list[str]] = {"project": [], "user": [], "builtin": []}

        project_dir = _project_templates_dir(self.project_dir)
        if project_dir and project_dir.is_dir():
            for f in sorted(project_dir.rglob("*")):
                if f.is_file():
                    layers["project"].append(str(f.relative_to(project_dir).as_posix()))

        user_dir = _user_templates_dir()
        if user_dir and user_dir.is_dir():
            for f in sorted(user_dir.rglob("*")):
                if f.is_file():
                    layers["user"].append(str(f.relative_to(user_dir).as_posix()))

        try:
            for f in _list_builtin_templates():
                layers["builtin"].append(f)
        except Exception as exc:
            logger.debug("Failed to list built-in templates: %s", exc, exc_info=True)

        return layers


def get_template_loader() -> TemplateLoader:
    """Return the shared :class:`TemplateLoader` singleton.

    Instantiates on first call, then reuses the same instance for every
    subsequent call.  This avoids repeated filesystem walks in
    :func:`find_project_config` when multiple modules create their own
    ``TemplateLoader()`` at import time.
    """
    if not _loader_instance:
        _loader_instance.append(TemplateLoader())
    return _loader_instance[0]


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}, text

    raw_yaml = match.group(1)
    body = text[match.end() :]

    try:
        fm: dict[str, Any] = yaml.safe_load(raw_yaml) or {}
    except Exception as exc:
        logger.debug("Failed to parse template frontmatter: %s", exc, exc_info=True)
        fm = {}

    return fm, body


# ── Path resolution ──────────────────────────────────────────────────


def _find_project_root() -> Path | None:
    toml_file = find_project_config()
    if toml_file is not None:
        return toml_file.parent
    return None


def _project_templates_dir(project_root: Path | None) -> Path | None:
    """Return the project-level templates directory path."""
    if project_root is None:
        return None
    return project_root / _PROJECT_TEMPLATES_DIR / _PROJECT_TEMPLATES_NAME


def _user_templates_dir() -> Path | None:
    """Return the user global templates directory path."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config_dir = Path(xdg) if xdg else Path.home() / ".config"
    candidate = config_dir / _USER_CONFIG_DIR / _USER_TEMPLATES_NAME
    return candidate if candidate.is_dir() else None


def _builtin_exists(template_file: str) -> bool:
    """Check if a built-in template exists."""
    try:
        _read_builtin(template_file)
        return True
    except Exception:
        return False


# ── Built-in template access (importlib.resources) ─────────────────


def _read_builtin(template_file: str) -> str:
    ref = ir.files(_BUILTIN_TEMPLATE_PACKAGE) / _BUILTIN_TEMPLATE_DIR / template_file
    return ref.read_text(encoding="utf-8")


def _list_builtin_templates() -> list[str]:
    tpl_dir = ir.files(_BUILTIN_TEMPLATE_PACKAGE) / _BUILTIN_TEMPLATE_DIR
    if not tpl_dir.is_dir():
        return []
    return sorted(str(p.relative_to(tpl_dir).as_posix()) for p in tpl_dir.rglob("*") if p.is_file())  # ty: ignore
