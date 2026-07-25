"""Tests for TemplateLoader with frontmatter parsing and variable validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from jinja2 import UndefinedError as Jinja2UndefinedError

from knox.core.templates import (
    TemplateLoader,
    TemplateMeta,
    parse_frontmatter,
)

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def project_dir(tmp_path: Any) -> Path:
    """Create a project with knox.toml and a templates dir."""
    root = tmp_path / "project"
    root.mkdir(parents=True)
    (root / "knox.toml").write_text("")
    return root


@pytest.fixture
def user_templates(tmp_path: Any, monkeypatch: Any) -> Path:
    """Set up a user global templates directory."""
    config_dir = tmp_path / "config"
    tpl_dir = config_dir / "knox" / "templates"
    tpl_dir.mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_dir))
    return tpl_dir


# ── Frontmatter parsing ─────────────────────────────────────────────────


class TestFrontmatterParsing:
    def test_no_frontmatter(self) -> None:
        text = "Hello {{ name }}!"
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert body == text

    def test_with_frontmatter(self) -> None:
        text = """---
variables:
  - name: title
    required: true
  - name: content
    required: false
role: lint.test
---

Page: {{ title }}
"""
        fm, body = parse_frontmatter(text)
        assert fm["role"] == "lint.test"
        assert len(fm["variables"]) == 2
        assert "Page: {{ title }}" in body

    def test_frontmatter_empty_body(self) -> None:
        text = "---\nrole: test\n---\n"
        fm, body = parse_frontmatter(text)
        assert fm["role"] == "test"
        assert body == ""

    def test_invalid_yaml_returns_empty(self) -> None:
        text = "---\n{{invalid: yaml: broken\n---\nbody"
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert "body" in body

    def test_no_leading_separator(self) -> None:
        text = "not frontmatter\n---\nbody"
        fm, body = parse_frontmatter(text)
        assert fm == {}  # no leading ---
        assert body == text


# ── TemplateMeta ────────────────────────────────────────────────────────


class TestTemplateMeta:
    def test_required_variable_names(self) -> None:
        meta = TemplateMeta(
            variables=[
                {"name": "title", "required": True},
                {"name": "content", "required": True},
                {"name": "optional_field", "required": False},
            ]
        )
        assert meta.required_variable_names() == {"title", "content"}

    def test_declared_variable_names(self) -> None:
        meta = TemplateMeta(
            variables=[
                {"name": "title"},
                {"name": "content"},
            ]
        )
        assert meta.declared_variable_names() == {"title", "content"}

    def test_from_yaml_block(self) -> None:
        yaml_text = "role: lint.test\nresponse_format: json_object\ntemperature: 0.5"
        meta = TemplateMeta.from_yaml_block(yaml_text)
        assert meta.role == "lint.test"
        assert meta.response_format == "json_object"
        assert meta.temperature == 0.5

    def test_from_yaml_block_variables(self) -> None:
        yaml_text = """
variables:
  - name: title
    required: true
  - name: content
    required: false
"""
        meta = TemplateMeta.from_yaml_block(yaml_text)
        assert len(meta.variables) == 2
        assert meta.required_variable_names() == {"title"}


# ── Validation ──────────────────────────────────────────────────────────


class TestValidation:
    def test_missing_required_variable_raises(self, project_dir: Path) -> None:
        tpl = project_dir / "knox" / "templates" / "req.md.j2"
        tpl.parent.mkdir(parents=True)
        tpl.write_text("---\nvariables:\n  - name: title\n    required: true\n---\nHello {{ title }}!")
        loader = TemplateLoader(project_dir=project_dir)
        with pytest.raises(ValueError, match="requires missing variables"):
            loader.render("req.md.j2")  # missing 'title'

    def test_all_required_variables_ok(self, project_dir: Path) -> None:
        tpl = project_dir / "knox" / "templates" / "req.md.j2"
        tpl.parent.mkdir(parents=True)
        tpl.write_text("---\nvariables:\n  - name: title\n    required: true\n---\nHello {{ title }}!")
        loader = TemplateLoader(project_dir=project_dir)
        result = loader.render("req.md.j2", title="World")
        assert result == "Hello World!"

    def test_strict_undefined_catches_missing_var(self, project_dir: Path) -> None:
        """No frontmatter, but template references {{ content }} without it being passed."""
        tpl = project_dir / "knox" / "templates" / "strict.md.j2"
        tpl.parent.mkdir(parents=True)
        tpl.write_text("Hello {{ name }}!")
        loader = TemplateLoader(project_dir=project_dir)
        with pytest.raises(Jinja2UndefinedError):
            loader.render("strict.md.j2")  # 'name' not passed

    def test_extra_unused_variable_warns(self, project_dir: Path, caplog: Any) -> None:
        tpl = project_dir / "knox" / "templates" / "simple.md.j2"
        tpl.parent.mkdir(parents=True)
        tpl.write_text("Hello {{ name }}!")
        loader = TemplateLoader(project_dir=project_dir)
        caplog.set_level(logging.WARNING)
        result = loader.render("simple.md.j2", name="Alice", unused_param="should warn")
        assert result == "Hello Alice!"
        assert any("unused_param" in record.message for record in caplog.records)

    def test_typed_variables_pass_validation(self, project_dir: Path) -> None:
        tpl = project_dir / "knox" / "templates" / "typed.md.j2"
        tpl.parent.mkdir(parents=True)
        tpl.write_text(
            "---\n"
            "variables:\n"
            "  - name: title\n"
            "    required: true\n"
            "  - name: content\n"
            "    required: true\n"
            "role: lint.test\n"
            "response_format: json_object\n"
            "temperature: 0.2\n"
            "---\n"
            "# {{ title }}\n\n{{ content }}"
        )
        loader = TemplateLoader(project_dir=project_dir)
        result = loader.render("typed.md.j2", title="Test", content="Hello")
        assert result == "# Test\n\nHello"
        meta = loader.get_meta("typed.md.j2")
        assert meta.role == "lint.test"
        assert meta.response_format == "json_object"
        assert meta.temperature == 0.2

    def test_get_meta_cached(self, project_dir: Path) -> None:
        tpl = project_dir / "knox" / "templates" / "meta.md.j2"
        tpl.parent.mkdir(parents=True)
        tpl.write_text("---\nrole: test\n---\n{{ x }}")
        loader = TemplateLoader(project_dir=project_dir)
        m1 = loader.get_meta("meta.md.j2")
        m2 = loader.get_meta("meta.md.j2")
        assert m1 is m2  # same cached object


# ── Basic loading (no frontmatter) ──────────────────────────────────────


class TestBasicLoading:
    def test_project_template_found(self, project_dir: Path) -> None:
        tpl = project_dir / "knox" / "templates" / "greeting.md.j2"
        tpl.parent.mkdir(parents=True)
        tpl.write_text("Hello {{ name }}!")
        loader = TemplateLoader(project_dir=project_dir)
        result = loader.render("greeting.md.j2", name="Alice")
        assert result == "Hello Alice!"

    def test_raw_text(self, project_dir: Path) -> None:
        tpl = project_dir / "knox" / "templates" / "greeting.md.j2"
        tpl.parent.mkdir(parents=True)
        tpl.write_text("Hello {{ name }}!")
        loader = TemplateLoader(project_dir=project_dir)
        text = loader.get_text("greeting.md.j2")
        assert "{{ name }}" in text

    def test_get_body_strips_frontmatter(self, project_dir: Path) -> None:
        tpl = project_dir / "knox" / "templates" / "body.md.j2"
        tpl.parent.mkdir(parents=True)
        tpl.write_text("---\nrole: test\n---\nJust the body {{ x }}")
        loader = TemplateLoader(project_dir=project_dir)
        body = loader.get_body("body.md.j2")
        assert "Just the body" in body
        assert "role: test" not in body

    def test_missing_template_raises(self, project_dir: Path) -> None:
        loader = TemplateLoader(project_dir=project_dir)
        with pytest.raises(FileNotFoundError, match="Template not found"):
            loader.get_text("nonexistent.md.j2")

    def test_subdir_template(self, project_dir: Path) -> None:
        sub = project_dir / "knox" / "templates" / "lint"
        sub.mkdir(parents=True)
        (sub / "test.md.j2").write_text("Score: {{ score }}")
        loader = TemplateLoader(project_dir=project_dir)
        result = loader.render("lint/test.md.j2", score=95)
        assert result == "Score: 95"

    def test_locate_source(self, project_dir: Path) -> None:
        tpl = project_dir / "knox" / "templates" / "greeting.md.j2"
        tpl.parent.mkdir(parents=True)
        tpl.write_text("Hello {{ name }}!")
        loader = TemplateLoader(project_dir=project_dir)
        source = loader.locate("greeting.md.j2")
        assert source.source == "project"


# ── User override ───────────────────────────────────────────────────────


class TestUserOverride:
    def test_user_template_used_when_no_project(self, tmp_path: Any, monkeypatch: Any) -> None:
        config_dir = tmp_path / "config"
        tpl_dir = config_dir / "knox" / "templates"
        tpl_dir.mkdir(parents=True)
        (tpl_dir / "hello.md.j2").write_text("User: {{ msg }}")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_dir))
        loader = TemplateLoader(project_dir=None)
        result = loader.render("hello.md.j2", msg="test")
        assert result == "User: test"

    def test_project_overrides_user(self, project_dir: Path, user_templates: Path) -> None:
        tpl = project_dir / "knox" / "templates" / "greeting.md.j2"
        tpl.parent.mkdir(parents=True)
        tpl.write_text("Project version: {{ name }}")
        (user_templates / "greeting.md.j2").write_text("User version: {{ name }}")
        loader = TemplateLoader(project_dir=project_dir)
        result = loader.render("greeting.md.j2", name="win")
        assert result == "Project version: win"

    def test_user_fallback_when_project_missing(self, project_dir: Path, user_templates: Path) -> None:
        (user_templates / "user_only.md.j2").write_text("User: {{ msg }}")
        loader = TemplateLoader(project_dir=project_dir)
        result = loader.render("user_only.md.j2", msg="fallback")
        assert result == "User: fallback"


# ── Caching ─────────────────────────────────────────────────────────────


class TestCaching:
    def test_cache_returns_same_result(self, project_dir: Path) -> None:
        tpl = project_dir / "knox" / "templates" / "greeting.md.j2"
        tpl.parent.mkdir(parents=True)
        tpl.write_text("Hi {{ name }}!")
        loader = TemplateLoader(project_dir=project_dir)
        r1 = loader.get_text("greeting.md.j2")
        r2 = loader.get_text("greeting.md.j2")
        assert r1 == r2

    def test_cache_recovers_after_delete(self, project_dir: Path) -> None:
        tpl = project_dir / "knox" / "templates" / "greeting.md.j2"
        tpl.parent.mkdir(parents=True)
        tpl.write_text("Hi {{ name }}!")
        loader = TemplateLoader(project_dir=project_dir)
        _ = loader.get_text("greeting.md.j2")
        tpl.unlink()
        with pytest.raises(FileNotFoundError):
            loader.get_text("greeting.md.j2")


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_render_with_empty_string_variable(self, project_dir: Path) -> None:
        tpl = project_dir / "knox" / "templates" / "empty.md.j2"
        tpl.parent.mkdir(parents=True)
        tpl.write_text("{{ x }}")
        loader = TemplateLoader(project_dir=project_dir)
        assert loader.render("empty.md.j2", x="") == ""

    def test_render_with_special_chars(self, project_dir: Path) -> None:
        tpl = project_dir / "knox" / "templates" / "special.md.j2"
        tpl.parent.mkdir(parents=True)
        tpl.write_text("{{ content }}")
        loader = TemplateLoader(project_dir=project_dir)
        result = loader.render("special.md.j2", content="<script>alert('xss')</script>")
        assert "<script>" in result  # autoescape is off

    def test_multi_line_template(self, project_dir: Path) -> None:
        tpl = project_dir / "knox" / "templates" / "multi.md.j2"
        tpl.parent.mkdir(parents=True)
        tpl.write_text("{% for item in items %}\n- {{ item }}\n{% endfor %}")
        loader = TemplateLoader(project_dir=project_dir)
        result = loader.render("multi.md.j2", items=["a", "b", "c"])
        assert result == "- a\n- b\n- c\n"
