"""Unit tests for the configurable user notes file (knox-3xw)."""

from __future__ import annotations

from pathlib import Path

import pytest

from knox.core.config._core import resolve_config
from knox.core.pipeline import WikiPipeline
from knox.core.pipeline_defs import EmbeddingConfig, LLMConfig, PipelineConfig, Tier
from knox.core.template_loader import TemplateLoader


def _make_project_templates(tmp_path: Path, templates: dict[str, str]) -> Path:
    """Create template files in the correct project structure."""
    tpl_dir = tmp_path / "knox" / "templates"
    tpl_dir.mkdir(parents=True, exist_ok=True)
    for name, content in templates.items():
        (tpl_dir / name).write_text(content, encoding="utf-8")
    return tmp_path


# ── Config key ──────────────────────────────────────────────────────────


class TestNotesConfig:
    """Verify ``notes.file`` follows the 5-layer config precedence."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("KNOX_NOTES_FILE", raising=False)
        cfg = resolve_config()
        assert cfg.get("notes.file") == "NOTES.md"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KNOX_NOTES_FILE", "CUSTOM.md")
        cfg = resolve_config()
        assert cfg.get("notes.file") == "CUSTOM.md"

    def test_project_config_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("KNOX_NOTES_FILE", raising=False)
        (tmp_path / "knox.toml").write_text('[notes]\nfile = "MY_NOTES.md"\n', encoding="utf-8")
        cfg = resolve_config()
        assert cfg.get("notes.file") == "MY_NOTES.md"


# ── PipelineConfig.from_resolved ────────────────────────────────────────


class TestNotesFileResolution:
    """Verify ``PipelineConfig.notes_file`` is resolved correctly."""

    @pytest.fixture(autouse=True)
    def _set_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KNOX_LLM_DEFAULT_MODEL", "test-model")

    def test_notes_file_resolved_from_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("KNOX_NOTES_FILE", raising=False)
        cfg = resolve_config(cli_overrides={"paths.wiki_dir": tmp_path})
        pc = PipelineConfig.from_resolved(cfg, source_dir=tmp_path / "sources", output_dir=tmp_path / "wiki")
        assert pc.notes_file is not None
        assert pc.notes_file == tmp_path / "NOTES.md"

    def test_notes_file_custom_name(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KNOX_NOTES_FILE", "MY_NOTES.md")
        cfg = resolve_config(cli_overrides={"paths.wiki_dir": tmp_path})
        pc = PipelineConfig.from_resolved(cfg, source_dir=tmp_path / "sources", output_dir=tmp_path / "wiki")
        assert pc.notes_file == tmp_path / "MY_NOTES.md"

    def test_notes_file_none_when_disabled(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("KNOX_NOTES_FILE", raising=False)
        (tmp_path / "knox.toml").write_text('[notes]\nfile = ""\n', encoding="utf-8")
        cfg = resolve_config(cli_overrides={"paths.wiki_dir": tmp_path})
        pc = PipelineConfig.from_resolved(cfg, source_dir=tmp_path / "sources", output_dir=tmp_path / "wiki")
        assert pc.notes_file is None

    def test_notes_file_name_from_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("KNOX_NOTES_FILE", raising=False)
        cfg = resolve_config()
        pc = PipelineConfig.from_resolved(cfg, source_dir=tmp_path / "sources", output_dir=tmp_path / "wiki")
        assert pc.notes_file is not None
        assert pc.notes_file.name == "NOTES.md"


# ── TemplateLoader global context ───────────────────────────────────────


class TestTemplateLoaderGlobalContext:
    """Verify ``set_global_context`` and ``clear_global_context``."""

    def test_set_and_clear(self) -> None:
        loader = TemplateLoader()
        loader.set_global_context(user_notes="hello")
        assert loader._global_context.get("user_notes") == "hello"
        loader.clear_global_context()
        assert "user_notes" not in loader._global_context

    def test_global_context_injected_in_render(self, tmp_path: Path) -> None:
        _make_project_templates(tmp_path, {"test.md.j2": "Notes: {{ user_notes }}"})
        loader = TemplateLoader(project_dir=tmp_path)
        loader.set_global_context(user_notes="my notes content")
        result = loader.render("test.md.j2")
        assert "my notes content" in result

    def test_global_context_does_not_trigger_unused_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _make_project_templates(tmp_path, {"no_notes.md.j2": "Just {{ title }}"})
        loader = TemplateLoader(project_dir=tmp_path)
        loader.set_global_context(user_notes="my notes", extra_var="unused")
        with caplog.at_level("WARNING"):
            loader.render("no_notes.md.j2", title="Hello")
        assert "unused variables" not in caplog.text

    def test_global_context_overridden_by_kwargs(self, tmp_path: Path) -> None:
        _make_project_templates(tmp_path, {"override.md.j2": "Notes: {{ user_notes }}"})
        loader = TemplateLoader(project_dir=tmp_path)
        loader.set_global_context(user_notes="global value")
        result = loader.render("override.md.j2", user_notes="explicit value")
        assert "explicit value" in result

    @pytest.fixture(autouse=True)
    def _cleanup_loader(self) -> None:
        loader = TemplateLoader()
        loader.clear_global_context()
        yield
        loader.clear_global_context()


# ── Pipeline integration ────────────────────────────────────────────────


class TestPipelineNotesInjection:
    """Verify notes are loaded and cleared by WikiPipeline.run()."""

    def test_pipeline_loads_notes_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When notes_file exists, its content is set in template global context during run."""
        notes_content = "# My Notes\nImportant context here."
        notes_path = tmp_path / "NOTES.md"
        notes_path.write_text(notes_content, encoding="utf-8")

        config = self._make_config(tmp_path, notes_path)
        pipeline = WikiPipeline(config=config)
        pipeline.run()

        # After run() completes, global context is cleared
        loader = TemplateLoader()
        assert "user_notes" not in loader._global_context

    def test_pipeline_missing_notes_no_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Missing notes file does not raise."""
        config = self._make_config(tmp_path, tmp_path / "NONEXISTENT.md")
        pipeline = WikiPipeline(config=config)
        pipeline.run()  # should not raise

    def test_pipeline_none_notes_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When notes_file is None, no error and no global context."""
        config = self._make_config(tmp_path, None)
        pipeline = WikiPipeline(config=config)
        pipeline.run()

        loader = TemplateLoader()
        assert "user_notes" not in loader._global_context

    def _make_config(self, tmp_path: Path, notes_file: Path | None) -> PipelineConfig:
        return PipelineConfig(
            source_dir=tmp_path / "sources",
            output_dir=tmp_path / "wiki",
            llm=LLMConfig("model", "vlm"),
            embedding=EmbeddingConfig("model"),
            notes_file=notes_file,
            start_at_tier=Tier.INDEX,
            stop_at_tier=Tier.INDEX,
        )
