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

logger = logging.getLogger(__name__)


TEMPLATES_PACKAGE: Final[str] = "knox.scaffolding.templates"

TEMPLATE_CONFIG: Final[str] = "config.toml"
TEMPLATE_NOTES: Final[str] = "NOTES.md"
TEMPLATE_README: Final[str] = "README.md"
TEMPLATE_GITIGNORE: Final[str] = "gitignore"

AUTHORED_TEMPLATES: Final[tuple[str, ...]] = (TEMPLATE_NOTES, TEMPLATE_README)

PROJECT_MARKERS: Final[tuple[str, ...]] = ("knox.toml", ".knox/config.toml")


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


def _existing_project_marker(target: Path) -> Path | None:
    """Return the first existing project marker in *target*, or ``None``."""
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
        or .knox/config.toml). Other files are left untouched.
    git:
        When ``True``, run ``git init`` in the project directory and
        create an initial commit.

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
    target = Path(target_dir).expanduser().resolve()

    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
    elif not target.is_dir():
        msg = f"Target path exists and is not a directory: {target}"
        raise ScaffoldError(msg)

    existing = _existing_project_marker(target)
    if existing is not None and not force:
        msg = f"Existing project detected ({existing}). Use --force to overwrite."
        raise ScaffoldError(msg)

    result = ScaffoldResult(target_dir=target)

    for relative in ("sources", "wiki", ".knox", ".knox/cache"):
        result.created_dirs.append(_create_directory(target, relative))

    config_relative = ".knox/config.toml"
    _copy_template(TEMPLATE_CONFIG, target, config_relative)
    result.copied_files.append(target / config_relative)

    _copy_template(TEMPLATE_GITIGNORE, target, ".gitignore")
    result.copied_files.append(target / ".gitignore")

    if bare:
        result.skipped_files.extend(AUTHORED_TEMPLATES)
    else:
        for template_name in AUTHORED_TEMPLATES:
            _copy_template(template_name, target, template_name)
            result.copied_files.append(target / template_name)

    if git:
        try:
            _initialise_git(target)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip()
            msg = f"git init failed: {stderr or exc}"
            raise ScaffoldError(msg) from exc
        result.git_initialised = True

    logger.info("Scaffolded knox wiki at %s", target)
    return result
