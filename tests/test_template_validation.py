"""Verify the project template was applied correctly."""

from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).parent.parent


def test_copier_answers_exists() -> None:
    answers = PROJECT_ROOT_DIR / ".copier-answers.yml"
    assert answers.exists(), "copier answers file missing"


def test_mise_config_structure() -> None:
    """Verify mise config in .config/mise/ (template style), not root mise.toml."""
    config_dir = PROJECT_ROOT_DIR / ".config" / "mise"
    assert config_dir.exists(), f"mise config dir missing at {config_dir}"

    config_toml = config_dir / "config.toml"
    assert config_toml.exists(), f"mise config.toml missing at {config_toml}"

    # Should have conf.d/ with task files
    conf_d = config_dir / "conf.d"
    assert conf_d.exists(), "mise conf.d missing"

    # Should have at least some task files
    toml_files = list(conf_d.glob("*.toml"))
    assert len(toml_files) >= 1, f"no task files in conf.d, found: {list(conf_d.iterdir())}"


def test_ruff_config_exists() -> None:
    ruff = PROJECT_ROOT_DIR / "ruff.toml"
    assert ruff.exists(), "ruff.toml missing"


def test_ty_config_exists() -> None:
    ty = PROJECT_ROOT_DIR / "ty.toml"
    assert ty.exists(), "ty.toml missing"
