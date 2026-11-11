"""Tests for the vault ``.gitignore``: rendering, merging, and the CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest

from kompend.cli.app import app
from kompend.core.paths import GITIGNORE_FILE
from kompend.scaffolding import (
    BLOCK_BEGIN,
    BLOCK_END,
    is_current,
    managed_block,
    merge_gitignore,
    render_gitignore,
)
from tests.helpers.cli import CliRunner

runner = CliRunner()


class TestRenderGitignore:
    def test_every_never_come_back_artifact_is_ignored(self) -> None:
        """Each rebuildable artifact family must be covered by a rule."""
        rendered = render_gitignore(sources_dir="sources", output_dir="wiki")
        for rule in (
            ".kompend/cache/",
            ".kompend/logs/",
            ".kompend/tmp/",
            ".kompend/vectors.db*",
            ".kompend/memory.db*",
            ".kompend/steps.db*",
            ".kompend/dedup/",
            "*.local.toml",
        ):
            assert rule in rendered, rule

    def test_the_store_and_config_are_not_ignored(self) -> None:
        """The only-copy data — the store and the config — stays commit-able."""
        rendered = render_gitignore(sources_dir="sources", output_dir="wiki")
        ignored = _rules(rendered)
        assert ".kompend/kompend.db" not in ignored
        assert ".kompend/config.toml" not in ignored
        assert "sources/" not in ignored

    def test_configured_directories_are_interpolated(self) -> None:
        rendered = render_gitignore(sources_dir="My Sources", output_dir="My Wiki")
        assert "My Wiki/" in rendered
        assert "My Sources/" in rendered
        assert "wiki/" not in _rules(rendered)

    def test_the_wiki_is_ignored_but_opt_in_is_documented(self) -> None:
        rendered = render_gitignore(sources_dir="sources", output_dir="wiki")
        assert "wiki/" in _rules(rendered)
        assert "# attachments/" in rendered  # commented opt-in, not a live rule

    def test_the_managed_block_is_delimited(self) -> None:
        rendered = render_gitignore(sources_dir="sources", output_dir="wiki")
        block = managed_block(rendered)
        assert block is not None
        assert block.startswith(BLOCK_BEGIN)
        assert block.rstrip("\n").endswith(BLOCK_END)


class TestMergeGitignore:
    def test_a_missing_file_gets_the_rendered_text(self) -> None:
        rendered = render_gitignore(sources_dir="sources", output_dir="wiki")
        assert merge_gitignore("", rendered) == rendered

    def test_user_rules_above_and_below_survive(self) -> None:
        rendered = render_gitignore(sources_dir="sources", output_dir="wiki")
        existing = f"*.pyc\n{managed_block(rendered)}\nmy-notes.txt\n"
        merged = merge_gitignore(existing, rendered)
        assert merged.startswith("*.pyc\n")
        assert merged.rstrip("\n").endswith("my-notes.txt")

    def test_a_stale_block_is_replaced_not_duplicated(self) -> None:
        rendered = render_gitignore(sources_dir="sources", output_dir="wiki")
        stale = f"# mine\n{BLOCK_BEGIN}\nold-rule\n{BLOCK_END}\n"
        merged = merge_gitignore(stale, rendered)
        assert "old-rule" not in merged
        assert merged.count(BLOCK_BEGIN) == 1
        assert is_current(merged, rendered)

    def test_a_file_without_a_block_gains_one_at_the_end(self) -> None:
        rendered = render_gitignore(sources_dir="sources", output_dir="wiki")
        merged = merge_gitignore("*.pyc\n", rendered)
        assert merged.startswith("*.pyc\n\n")
        assert is_current(merged, rendered)

    def test_a_rendered_text_without_a_block_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="no managed block"):
            merge_gitignore("", "plain rules only\n")


class TestIsCurrent:
    def test_identical_blocks_are_current(self) -> None:
        rendered = render_gitignore(sources_dir="sources", output_dir="wiki")
        assert is_current(rendered, rendered)

    def test_a_different_block_is_stale(self) -> None:
        rendered = render_gitignore(sources_dir="sources", output_dir="wiki")
        other = render_gitignore(sources_dir="sources", output_dir="other")
        assert not is_current(other, rendered)

    def test_a_file_without_a_block_is_stale(self) -> None:
        rendered = render_gitignore(sources_dir="sources", output_dir="wiki")
        assert not is_current("*.pyc\n", rendered)


class TestGitignoreCommand:
    def test_creates_the_file(self, tmp_path: Path) -> None:
        vault = self._vault(tmp_path)
        result = runner.invoke(app, ["-w", str(vault), "gitignore"])
        assert result.exit_code == 0, result.stdout + result.stderr
        text = (vault / GITIGNORE_FILE).read_text(encoding="utf-8")
        assert managed_block(text) is not None
        assert is_current(text, render_gitignore(sources_dir="sources", output_dir="wiki"))

    def test_keeps_the_user_rules(self, tmp_path: Path) -> None:
        vault = self._vault(tmp_path)
        path = vault / GITIGNORE_FILE
        path.write_text("*.pyc\n", encoding="utf-8")
        result = runner.invoke(app, ["-w", str(vault), "gitignore"])
        assert result.exit_code == 0, result.stdout + result.stderr
        text = path.read_text(encoding="utf-8")
        assert text.startswith("*.pyc\n")
        assert managed_block(text) is not None

    def test_is_idempotent(self, tmp_path: Path) -> None:
        vault = self._vault(tmp_path)
        runner.invoke(app, ["-w", str(vault), "gitignore"])
        first = (vault / GITIGNORE_FILE).read_text(encoding="utf-8")
        result = runner.invoke(app, ["-w", str(vault), "gitignore"])
        assert result.exit_code == 0
        assert (vault / GITIGNORE_FILE).read_text(encoding="utf-8") == first
        assert "already up to date" in result.stdout

    def test_check_reports_a_current_file(self, tmp_path: Path) -> None:
        vault = self._vault(tmp_path)
        runner.invoke(app, ["-w", str(vault), "gitignore"])
        result = runner.invoke(app, ["-w", str(vault), "gitignore", "--check"])
        assert result.exit_code == 0, result.stdout + result.stderr

    def test_check_fails_on_a_missing_file(self, tmp_path: Path) -> None:
        vault = self._vault(tmp_path)
        result = runner.invoke(app, ["-w", str(vault), "gitignore", "--check"])
        assert result.exit_code == 1
        assert "does not exist" in result.stderr

    def test_check_fails_on_a_stale_block(self, tmp_path: Path) -> None:
        vault = self._vault(tmp_path)
        (vault / GITIGNORE_FILE).write_text(f"{BLOCK_BEGIN}\nold\n{BLOCK_END}\n", encoding="utf-8")
        result = runner.invoke(app, ["-w", str(vault), "gitignore", "--check"])
        assert result.exit_code == 1
        assert "outdated" in result.stderr

    def test_print_does_not_touch_the_vault(self, tmp_path: Path) -> None:
        vault = self._vault(tmp_path)
        result = runner.invoke(app, ["-w", str(vault), "gitignore", "--print"])
        assert result.exit_code == 0
        assert BLOCK_BEGIN in result.stdout
        assert not (vault / GITIGNORE_FILE).exists()

    def _vault(self, tmp_path: Path) -> Path:
        """A minimal vault: kompend.toml marks the root, sources/ is the input."""
        (tmp_path / "kompend.toml").write_text("# vault\n", encoding="utf-8")
        (tmp_path / "sources").mkdir()
        return tmp_path


def _rules(text: str) -> set[str]:
    """Live (uncommented) rules in *text*, trimmed to their bare patterns."""
    return {line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")}
