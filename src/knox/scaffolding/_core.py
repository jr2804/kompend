"""Scaffolding utilities for KNOX wiki projects.

The :func:`scaffold_project` function creates the standard directory
layout and copies template files for a new wiki project. It is
consumed by the ``knox init`` CLI command.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Final

from knox.core.config._core import DEFAULTS
from knox.core.paths import CACHE_DIR, CACHE_SUBDIR, CONFIG_FILE, KNOX_CONFIG_FILE, KNOX_DIR, PROJECT_CONFIG_FILE

logger = logging.getLogger(__name__)


TEMPLATES_PACKAGE: Final[str] = "knox.scaffolding.templates"

TEMPLATE_CONFIG: Final[str] = CONFIG_FILE + ".j2"
TEMPLATE_NOTES: Final[str] = "NOTES.md"
TEMPLATE_README: Final[str] = "README.md"
TEMPLATE_GITIGNORE: Final[str] = "gitignore"

AUTHORED_TEMPLATES: Final[tuple[str, ...]] = (TEMPLATE_NOTES, TEMPLATE_README)

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


class ScaffoldError(Exception):
    """Raised when scaffolding cannot proceed (e.g. target dir conflicts)."""


def _template_path(name: str) -> Path:
    """Return the filesystem path of a template file shipped with the package."""
    return Path(str(resources.files(TEMPLATES_PACKAGE).joinpath(name)))


def find_existing_project(target: Path) -> Path | None:
    """Return the first existing project marker in *target*, or ``None``.

    A project marker is either ``knox.toml`` (project-level config) or
    ``.knox/config.toml`` (vault-level config). The returned path is the
    marker file that was found.
    """
    for marker in PROJECT_MARKERS:
        candidate = target / marker
        if candidate.exists():
            return candidate
    return None


def _create_directory(target: Path, relative: str) -> Path:
    """Create a directory under *target* and record it in *result*."""
    new_dir = target / relative
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir


def _copy_template(template_name: str, target: Path, relative: str) -> None:
    """Copy a template file into *target/relative*."""
    source = _template_path(template_name)
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _clear_directory(target: Path, *, skip: str | None = None) -> None:
    """Remove all contents of *target*, optionally skipping a named child."""
    for item in target.iterdir():
        if skip is not None and item.name == skip:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _render_config_template(target_dir: Path, relative: str) -> None:
    """Render the Jinja2 config template with current :data:`DEFAULTS`.

    Reads ``config.toml.j2`` from the templates package, renders it
    with the ``defaults`` variable set to :data:`DEFAULTS`, and writes
    the result to *target_dir/relative*.  This keeps config values in
    sync with code — no more hard-coded duplicates.
    """
    from jinja2 import Template

    source = _template_path(TEMPLATE_CONFIG)
    destination = target_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)

    raw = source.read_text(encoding="utf-8")
    rendered = Template(raw).render(defaults=DEFAULTS)
    destination.write_text(rendered, encoding="utf-8")
    logger.info("Rendered config template to %s", destination)


def _initialise_git(target: Path) -> None:
    """Run ``git init`` in *target* and stage an initial commit."""
    git_path = shutil.which("git")
    if not git_path:
        raise FileNotFoundError("git executable not found in PATH")

    # All arguments are hard-coded constants — no untrusted input reaches subprocess.
    subprocess.run(  # noqa: S603 -- trusted executable with fixed args
        [git_path, "init", "--initial-branch=main"],
        cwd=target,
        check=True,
        capture_output=True,
    )
    subprocess.run(  # noqa: S603 -- trusted executable with fixed args
        [git_path, "add", "."],
        cwd=target,
        check=True,
        capture_output=True,
    )
    subprocess.run(  # noqa: S603 -- trusted executable with fixed args
        [git_path, "-c", "user.name=knox init", "-c", "user.email=knox@local", "commit", "-m", "chore: initialise knox wiki project"],
        cwd=target,
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
        the target directory before scaffolding — except
        ``.knox/cache/`` which is preserved in place. This enables fast
        re-runs where ingest can skip already-parsed documents.

    Returns:
    -------
    ScaffoldResult
        A summary of what was created.

    Raises:
    ------
    ScaffoldError
        If the target directory contains an existing project marker and
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

    # When force + keep_cache: clear everything except .knox/cache/.
    if force and keep_cache:
        knox_path = resolved_target_dir / KNOX_DIR
        _clear_directory(resolved_target_dir, skip=KNOX_DIR)
        if knox_path.exists():
            _clear_directory(knox_path, skip=CACHE_SUBDIR)

    result = ScaffoldResult(target_dir=resolved_target_dir)

    sources_dir_name = DEFAULTS["paths"]["sources_dir"]
    output_dir_name = DEFAULTS["paths"]["output_dir"]
    for relative in (
        sources_dir_name,
        output_dir_name,
        KNOX_DIR,
        CACHE_DIR,
    ):
        result.created_dirs.append(_create_directory(resolved_target_dir, relative))

    config_relative_file = KNOX_CONFIG_FILE
    # Render the Jinja2 config template with current DEFAULTS so values
    # are always derived from code — never hard-coded in the template.
    _render_config_template(resolved_target_dir, config_relative_file)
    result.copied_files.append(resolved_target_dir / config_relative_file)

    _copy_template(TEMPLATE_GITIGNORE, resolved_target_dir, ".gitignore")
    result.copied_files.append(resolved_target_dir / ".gitignore")

    if bare:
        result.skipped_files.extend(AUTHORED_TEMPLATES)
    else:
        for template_name in AUTHORED_TEMPLATES:
            _copy_template(template_name, resolved_target_dir, template_name)
            result.copied_files.append(resolved_target_dir / template_name)

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
