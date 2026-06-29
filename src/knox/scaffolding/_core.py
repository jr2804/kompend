"""Scaffolding utilities for KNOX wiki projects.

The :func:`scaffold_project` function creates the standard directory
layout and copies template files for a new wiki project. It is
consumed by the ``knox init`` CLI command.
"""

from __future__ import annotations

import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Final

import jinja2

from knox.core.config._core import DEFAULTS
from knox.core.config.generate import generate_scaffold_config
from knox.core.paths import CACHE_DIR, CACHE_SUBDIR, DEDUP_DIR, KNOX_CONFIG_FILE, KNOX_DIR, PROJECT_CONFIG_FILE
from knox.logging import get_logger

logger = get_logger(__name__)


TEMPLATES_PACKAGE: Final[str] = "knox.scaffolding.templates"

TEMPLATE_NOTES: Final[str] = "NOTES.md"  # copied verbatim (contains user-facing {{ user_notes }})
TEMPLATE_README: Final[str] = "README.md.j2"  # rendered with config paths
TEMPLATE_GITIGNORE: Final[str] = "gitignore.j2"  # rendered with config paths
TEMPLATE_NEXT_STEPS: Final[str] = "next_steps.txt.j2"  # rendered with config paths (console output only)

AUTHORED_TEMPLATES: Final[tuple[str, ...]] = ("NOTES.md", "README.md")

PROJECT_MARKERS: Final[tuple[str, ...]] = (PROJECT_CONFIG_FILE, KNOX_CONFIG_FILE)


@dataclass
class ScaffoldResult:
    """Summary of a scaffolding run.

    Attributes:
    ----------
    target_dir:
        The resolved absolute path of the project directory.
    created_dirs:
        Directories that were created during this run.
    copied_files:
        Files that were copied from templates (relative to ``target_dir``).
    skipped_files:
        Files that were intentionally not created (e.g. ``--bare``).
    git_initialised:
        Whether ``git init`` was run (only with ``--git``).
    """

    target_dir: Path
    created_dirs: list[Path] = field(default_factory=list)
    copied_files: list[Path] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    git_initialised: bool = False
    sources_dir: str = ""
    output_dir: str = ""


class ScaffoldError(Exception):
    """Raised when scaffolding cannot proceed (e.g. target_dir dir conflicts)."""


def _render_template(template_name: str, **vars: object) -> str:
    """Read a Jinja2 scaffold template and render it with *vars*.

    ``keep_trailing_newline=True`` preserves the trailing newline so
    rendered files match their source layout exactly.
    """
    content = _template_path(template_name).read_text(encoding="utf-8")
    return jinja2.Template(content, keep_trailing_newline=True).render(**vars)


def _template_path(name: str) -> Path:
    """Return the filesystem path of a template file shipped with the package."""
    return Path(str(resources.files(TEMPLATES_PACKAGE).joinpath(name)))


def find_existing_project(target_dir: Path) -> Path | None:
    """Return the first existing project marker in *target_dir*, or ``None``.

    A project marker is either ``knox.toml`` (project-level config) or
    ``.knox/config.toml`` (vault-level config). The returned path is the
    marker file that was found.
    """
    for marker in PROJECT_MARKERS:
        candidate = target_dir / marker
        if candidate.exists():
            return candidate
    return None


def _read_path_dirs(config_path: Path) -> tuple[str, str]:
    """Read ``paths.sources_dir`` and ``paths.output_dir`` from a config file.

    Falls back to :data:`DEFAULTS` when the file is missing or omits the
    keys. This is used by scaffold so created directories match the config
    being installed, not hardcoded defaults.
    """
    sources = DEFAULTS["paths"]["sources_dir"]
    output = DEFAULTS["paths"]["output_dir"]
    if config_path.is_file():
        try:
            data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError):
            return sources, output
        paths = data.get("paths", {})
        if isinstance(paths.get("sources_dir"), str):
            sources = paths["sources_dir"]
        if isinstance(paths.get("output_dir"), str):
            output = paths["output_dir"]
    return sources, output


def _create_directory(target_dir: Path, relative: str) -> Path:
    """Create a directory under *target_dir* and record it in *result*."""
    new_dir = target_dir / relative
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir


def _copy_template(template_name: str, target_dir: Path, relative: str) -> None:
    """Copy a template file into *target_dir/relative*."""
    source = _template_path(template_name)
    destination = target_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _clear_directory(target_dir: Path, *, skip: str | set[str] | None = None) -> None:
    """Remove all contents of *target_dir*, optionally skipping named children."""
    skip_set = {skip} if isinstance(skip, str) else (skip or set())
    for item in target_dir.iterdir():
        if item.name in skip_set:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _write_config(target_dir: Path, relative: str, config_file: Path | None = None) -> None:
    """Write the project config at ``target_dir / relative``.

    When *config_file* is given, it is copied verbatim — this is how
    ``knox init --config <path>`` installs a chosen config into a new vault
    (e.g. a cloud/local demo profile). Otherwise a config is generated from
    :data:`DEFAULTS` so values stay derived from code.
    """
    destination = target_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if config_file is not None:
        destination.write_text(Path(config_file).read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("Copied config from %s to %s", config_file, destination)
    else:
        destination.write_text(generate_scaffold_config(), encoding="utf-8")
        logger.info("Generated config at %s", destination)


def _initialise_git(target_dir: Path) -> None:
    """Run ``git init`` in *target_dir* and stage an initial commit."""
    git_path = shutil.which("git")
    if not git_path:
        raise FileNotFoundError("git executable not found in PATH")

    # All arguments are hard-coded constants — no untrusted input reaches subprocess.
    subprocess.run(  # noqa: S603 -- trusted executable with fixed args
        [git_path, "init", "--initial-branch=main"],
        cwd=target_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(  # noqa: S603 -- trusted executable with fixed args
        [git_path, "add", "."],
        cwd=target_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(  # noqa: S603 -- trusted executable with fixed args
        [git_path, "-c", "user.name=knox init", "-c", "user.email=knox@local", "commit", "-m", "chore: initialise knox wiki project"],
        cwd=target_dir,
        check=True,
        capture_output=True,
    )


def scaffold_project(
    target_dir: str | Path,
    *,
    bare: bool = False,
    force: bool = False,
    git: bool = False,
    keep_cache: bool = False,
    config_file: str | Path | None = None,
) -> ScaffoldResult:
    """Create a new KNOX wiki project in *target_dir*.

    Parameters
    ----------
    target_dir:
        Path to the project directory. Created if it does not exist.
    bare:
        When ``True``, skip authoring templates (``NOTES.md`` and
        ``README.md``). Runtime directories and configuration are still
        created.
    force:
        When ``True``, overwrite an existing project marker (knox.toml
        or .knox/config.toml). Other files are left untouched unless
        *keep_cache* is also ``True``.
    git:
        When ``True``, run ``git init`` in the project directory and
        create an initial commit.
    keep_cache:
        When ``True`` (requires *force*), remove **all** contents of
        the target_dir directory before scaffolding — except
        ``.knox/cache/`` which is preserved in place. This enables fast
        re-runs where ingest can skip already-parsed documents.
    config_file:
        Optional path to a TOML config file to install as the vault's
        ``.knox/config.toml`` instead of the generated default. Use this to
        seed a vault with a chosen LLM/budget/profile (e.g. a demo config).
        The file is copied verbatim; it must be valid KNOX TOML.

    Returns:
    -------
    ScaffoldResult
        A summary of what was created.

    Raises:
    ------
    ScaffoldError
        If the target_dir directory contains an existing project marker and
        ``force`` is not set, or if a filesystem operation fails.
    """
    resolved_target_dir = Path(target_dir).expanduser().resolve()

    if not resolved_target_dir.exists():
        resolved_target_dir.mkdir(parents=True, exist_ok=True)
    elif not resolved_target_dir.is_dir():
        msg = f"Target path exists and is not a directory: {resolved_target_dir}"
        raise ScaffoldError(msg)

    existing = find_existing_project(resolved_target_dir)
    if existing is not None and not force:
        msg = f"Existing project detected ({existing}). Use --force to overwrite."
        raise ScaffoldError(msg)

    # When force + keep_cache: clear everything except .knox/cache/ and .knox/dedup/.
    if force and keep_cache:
        knox_path = resolved_target_dir / KNOX_DIR
        _clear_directory(resolved_target_dir, skip=KNOX_DIR)
        if knox_path.exists():
            _clear_directory(knox_path, skip={CACHE_SUBDIR, DEDUP_DIR})

    result = ScaffoldResult(target_dir=resolved_target_dir)

    # Write the config FIRST so created directories match the config being
    # installed (paths.sources_dir / paths.output_dir may differ from DEFAULTS
    # when --config installs a custom profile). Previously these came from
    # DEFAULTS, so a non-default sources_dir left stale/empty directories.
    config_relative_file = KNOX_CONFIG_FILE
    _write_config(resolved_target_dir, config_relative_file, config_file=config_file)
    result.copied_files.append(resolved_target_dir / config_relative_file)

    sources_dir_name, output_dir_name = _read_path_dirs(resolved_target_dir / config_relative_file)
    result.sources_dir = sources_dir_name
    result.output_dir = output_dir_name
    render_vars = {"sources_dir": sources_dir_name, "output_dir": output_dir_name}
    for relative in (
        sources_dir_name,
        output_dir_name,
        KNOX_DIR,
        CACHE_DIR,
    ):
        result.created_dirs.append(_create_directory(resolved_target_dir, relative))

    # Render .gitignore from config paths (output_dir must be gitignored).
    gitignore_dest = resolved_target_dir / ".gitignore"
    gitignore_dest.write_text(_render_template(TEMPLATE_GITIGNORE, **render_vars), encoding="utf-8")
    result.copied_files.append(gitignore_dest)

    if bare:
        result.skipped_files.extend(AUTHORED_TEMPLATES)
    else:
        # NOTES.md is copied verbatim — it contains user-facing {{ user_notes }}
        # guidance and must not be rendered by Jinja2.
        _copy_template(TEMPLATE_NOTES, resolved_target_dir, "NOTES.md")
        result.copied_files.append(resolved_target_dir / "NOTES.md")
        # README.md is rendered with config paths.
        readme_dest = resolved_target_dir / "README.md"
        readme_dest.write_text(_render_template(TEMPLATE_README, **render_vars), encoding="utf-8")
        result.copied_files.append(readme_dest)

    if git:
        try:
            _initialise_git(resolved_target_dir)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip()
            msg = f"git init failed: {stderr or exc}"
            raise ScaffoldError(msg) from exc
        result.git_initialised = True

    logger.info("Scaffolded knox wiki at %s", resolved_target_dir)
    return result
