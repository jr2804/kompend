"""Scaffolding utilities for KOMPEND wiki projects.

The :func:`scaffold_project` function creates the standard directory
layout and copies template files for a new wiki project. It is
consumed by the ``kompend init`` CLI command.
"""

from __future__ import annotations

import shutil
import subprocess
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Final

import jinja2

from kompend.core.config._core import DEFAULTS
from kompend.core.config.generate import generate_scaffold_config
from kompend.core.paths import CACHE_DIR, CACHE_SUBDIR, DEDUP_DIR, KOMP_CONFIG_FILE, KOMP_DIR, PROJECT_CONFIG_FILE
from kompend.logging import get_logger

TEMPLATES_PACKAGE: Final[str] = "kompend.scaffolding.templates"

TEMPLATE_NOTES: Final[str] = "NOTES.md"  # copied verbatim (contains user-facing {{ user_notes }})
TEMPLATE_README: Final[str] = "README.md.j2"  # rendered with config paths
TEMPLATE_GITIGNORE: Final[str] = "gitignore.j2"  # rendered with config paths
TEMPLATE_NEXT_STEPS: Final[str] = "next_steps.txt.j2"  # rendered with config paths (console output only)
TEMPLATE_AGENTS: Final[str] = "AGENTS.md.j2"  # rendered with config paths (agent instructions)

AUTHORED_TEMPLATES: Final[tuple[str, ...]] = ("NOTES.md", "README.md")

#: Files at the project root this scaffold owns. Already-existing copies
#: of these are tracked content (.gitignore is the rail; AGENTS.md is the
#: DOX contract) and must not be clobbered silently. force=True is
#: required to overwrite them, and the clobbered set is reported so the
#: caller can warn. NOTES.md is the user-authored equivalent: the
#: scaffold only writes it in default mode (not bare), and an existing
#: file is treated the same way.
RAIL_FILES: Final[tuple[str, ...]] = (".gitignore", "AGENTS.md", "README.md")

PROJECT_MARKERS: Final[tuple[str, ...]] = (PROJECT_CONFIG_FILE, KOMP_CONFIG_FILE)

logger = get_logger(__name__)


@dataclass
class ScaffoldResult:
    """Summary of a scaffolding run.

    Attributes:
    ----------
    clobbered:
        Rail files (.gitignore, AGENTS.md, README.md) or NOTES.md that
        existed at the target *and* were overwritten by this run
        (populated only when force=True is passed). Surfaces the cost
        of --force so the caller can warn.

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
    #: Files this scaffold would have written that existed at the target
    #: already. Populated only when force=True overwrote them, so the CLI
    #: can warn.
    clobbered: list[Path] = field(default_factory=list)


class ScaffoldError(Exception):
    """Raised when scaffolding cannot proceed (e.g. target_dir dir conflicts)."""


def scaffold_project(
    target_dir: str | Path,
    *,
    bare: bool = False,
    force: bool = False,
    git: bool = False,
    keep_cache: bool = False,
    config_file: str | Path | None = None,
) -> ScaffoldResult:
    """Create a new KOMPEND wiki project in *target_dir*.

    Parameters
    ----------
    target_dir:
        Path to the project directory. Created if it does not exist.
    bare:
        When ``True``, skip authoring templates (``NOTES.md`` and
        ``README.md``). Runtime directories and configuration are still
        created.
    force:
        When ``True``, overwrite an existing project marker (kompend.toml
        or .kompend/config.toml) and any rail file the scaffold would
        otherwise write. Otherwise the run refuses to clobber an existing
        tracked rail file (.gitignore, AGENTS.md, README.md, NOTES.md) so
        the user gets a single, explicit refusal listing every offender.
    git:
        When ``True``, run ``git init`` in the project directory and
        create an initial commit.
    keep_cache:
        When ``True`` (requires *force*), remove **all** contents of
        the target_dir directory before scaffolding — except
        ``.kompend/cache/`` which is preserved in place. This enables fast
        re-runs where ingest can skip already-parsed documents.
    config_file:
        Optional path to a TOML config file to install as the vault's
        ``.kompend/config.toml`` instead of the generated default. Use this to
        seed a vault with a chosen LLM/budget/profile (e.g. a demo config).
        The file is copied verbatim; it must be valid KOMPEND TOML.

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

    # When force + keep_cache: clear everything except .kompend/cache/ and .kompend/dedup/.
    if force and keep_cache:
        kompend_dir = resolved_target_dir / KOMP_DIR
        _clear_directory(resolved_target_dir, skip=KOMP_DIR)
        if kompend_dir.exists():
            _clear_directory(kompend_dir, skip={CACHE_SUBDIR, DEDUP_DIR})

    result = ScaffoldResult(target_dir=resolved_target_dir)

    # Write the config FIRST so created directories match the config being
    # installed (paths.sources_dir / paths.output_dir may differ from DEFAULTS
    # when --config installs a custom profile). Previously these came from
    # DEFAULTS, so a non-default sources_dir left stale/empty directories.
    config_relative_file = KOMP_CONFIG_FILE
    _write_config(resolved_target_dir, config_relative_file, config_file=Path(config_file) if config_file else None)
    result.copied_files.append(resolved_target_dir / config_relative_file)

    # Two-phase rail-file guard. Phase 1 computes the set of files this
    # scaffold run *would* write. Phase 2 inspects every entry the scaffold
    # may produce and refuses the whole run if any of them pre-exists and is
    # not scaffold-owned — with a single error listing every offender. A rail
    # file is "scaffold-owned" iff the scaffold has already produced it in
    # this same call, so re-writing it (e.g. across bare/notes branches) is
    # idempotent and needs no separate check. NOTES.md is always checked,
    # even under --bare — a user-owned NOTES.md is content, and the scaffold
    # must not destroy it; --bare only stops the scaffold from writing it.
    files_this_run: set[Path] = {resolved_target_dir / config_relative_file}

    sources_dir_name, output_dir_name = _read_path_dirs(resolved_target_dir / config_relative_file)
    result.sources_dir = sources_dir_name
    result.output_dir = output_dir_name
    render_vars = {"sources_dir": sources_dir_name, "output_dir": output_dir_name}
    for relative in (
        sources_dir_name,
        output_dir_name,
        KOMP_DIR,
        CACHE_DIR,
    ):
        result.created_dirs.append(_create_directory(resolved_target_dir, relative))

    # Plan every file the scaffold may write. Each entry is (relative_path, writer).
    # NOTES.md is always checked: under --bare the scaffold does not write it,
    # but an existing user-owned NOTES.md still triggers a refusal. We split
    # the two concerns: every planned write is checked + written; NOTES.md gets
    # an additional check-only entry so --bare still guards the user's copy.
    planned: list[tuple[str, Callable[[], object]]] = [
        (
            ".gitignore",
            lambda: resolved_target_dir.joinpath(".gitignore").write_text(_render_template(TEMPLATE_GITIGNORE, **render_vars), encoding="utf-8"),
        ),
    ]
    if not bare:
        planned.append(
            (
                "README.md",
                lambda: resolved_target_dir.joinpath("README.md").write_text(_render_template(TEMPLATE_README, **render_vars), encoding="utf-8"),
            )
        )
        planned.append(
            (
                "NOTES.md",
                lambda: _copy_template(TEMPLATE_NOTES, resolved_target_dir, "NOTES.md"),
            )
        )
    planned.append(
        (
            "AGENTS.md",
            lambda: resolved_target_dir.joinpath("AGENTS.md").write_text(_render_template(TEMPLATE_AGENTS, **render_vars), encoding="utf-8"),
        )
    )

    # Add a check-only entry for NOTES.md even under --bare: the scaffold does
    # not write it, but the refusal must still fire if the user already has a
    # NOTES.md (otherwise --bare would silently let a scaffold destroy it).
    notes_check_relative = "NOTES.md" if bare or not any(r == "NOTES.md" for r, _ in planned) else None

    # Phase 1: check every planned write + the extra NOTES.md check. Collecting
    # first and raising once is the only way the user sees every offender, not
    # just the first.
    checked: list[tuple[str, Path, Callable[[], object]]] = []
    offenders: list[Path] = []
    for relative, writer in planned:
        dest = _check_rail_file_overwrite(resolved_target_dir, relative, files_this_run, force, result)
        if dest is None:
            offenders.append(resolved_target_dir / relative)
        else:
            checked.append((relative, dest, writer))
    if notes_check_relative is not None:
        dest = _check_rail_file_overwrite(resolved_target_dir, notes_check_relative, files_this_run, force, result)
        if dest is None:
            offenders.append(resolved_target_dir / notes_check_relative)
    if offenders:
        joined = ", ".join(repr(str(o)) for o in offenders)
        msg = f"refusing to overwrite existing tracked file(s) in {resolved_target_dir}: {joined}. Use --force to overwrite."
        raise ScaffoldError(msg)

    # Phase 2: every planned write is now approved; execute in order.
    for _relative, dest, writer in checked:
        writer()
        result.copied_files.append(dest)
        files_this_run.add(dest)

    # In --bare, NOTES.md and README.md are not scaffold-owned and not written
    # this run. Surface that as a skipped-file entry, mirroring the pre-refactor
    # contract that `kompend init` reports what the run did *not* do.
    if bare:
        result.skipped_files.extend(AUTHORED_TEMPLATES)

    if git:
        try:
            _initialise_git(resolved_target_dir)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip()
            msg = f"git init failed: {stderr or exc}"
            raise ScaffoldError(msg) from exc
        result.git_initialised = True

    logger.info("Scaffolded kompend wiki at %s", resolved_target_dir)
    return result


def _render_template(template_name: str, **vars: Any) -> str:
    """Read a Jinja2 scaffold template and render it with *vars*.

    ``keep_trailing_newline=True`` preserves the trailing newline so
    rendered files match their source layout exactly.
    """
    content = _template_file(template_name).read_text(encoding="utf-8")
    return jinja2.Template(content, keep_trailing_newline=True).render(**vars)


def find_existing_project(target_dir: Path) -> Path | None:
    """Return the first existing project marker in *target_dir*, or ``None``.

    A project marker is either ``kompend.toml`` (project-level config) or
    ``.kompend/config.toml`` (vault-level config). The returned path is the
    marker file that was found.
    """
    for marker in PROJECT_MARKERS:
        candidate = target_dir / marker
        if candidate.exists():
            return candidate
    return None


def _read_path_dirs(config_file: Path) -> tuple[str, str]:
    """Read ``paths.sources_dir`` and ``paths.output_dir`` from a config file.

    Falls back to :data:`DEFAULTS` when the file is missing or omits the
    keys. This is used by scaffold so created directories match the config
    being installed, not hardcoded defaults.
    """
    sources = DEFAULTS["paths"]["sources_dir"]
    output = DEFAULTS["paths"]["output_dir"]
    if config_file.is_file():
        try:
            data = tomllib.loads(config_file.read_text(encoding="utf-8"))
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
    source = _template_file(template_name)
    destination = target_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _template_file(name: str) -> Path:
    """Return the filesystem path of a template file shipped with the package."""
    return Path(str(resources.files(TEMPLATES_PACKAGE).joinpath(name)))


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
    ``kompend init --config <path>`` installs a chosen config into a new vault
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


def _check_rail_file_overwrite(
    target_dir: Path,
    relative: str,
    files_this_run: set[Path],
    force: bool,
    result: ScaffoldResult,
) -> Path | None:
    """Decide whether *target_dir / relative* may be written this run.

    Returns the destination Path if it may be written (either the scaffold
    already produced it in this call, or it does not exist yet, or force=True
    accepts the clobber). Returns ``None`` if the file exists, is not
    scaffold-owned, and force is off — in which case *target_dir / relative*
    is *not* recorded in ``result.clobbered`` (the caller collects the
    candidate for the final refusal and the destination stays untouched).

    The caller is responsible for *all* the rail files: collecting the
    candidates and raising a single error after the fact. This avoids a
    short-circuit on the first offender (which would hide the others).
    """
    dest = target_dir / relative
    if dest.exists() and dest not in files_this_run:
        if force:
            logger.warning("Scaffold is overwriting existing %r in %s", relative, target_dir)
            result.clobbered.append(dest)
        else:
            return None
    return dest


def _initialise_git(target_dir: Path) -> None:
    """Run ``git init`` in *target_dir* and stage an initial commit."""
    git_file = shutil.which("git")
    if not git_file:
        raise FileNotFoundError("git executable not found in PATH")

    # All arguments are hard-coded constants — no untrusted input reaches subprocess.
    subprocess.run(  # noqa: S603 -- trusted executable with fixed args
        [git_file, "init", "--initial-branch=main"],
        cwd=target_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(  # noqa: S603 -- trusted executable with fixed args
        [git_file, "add", "."],
        cwd=target_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(  # noqa: S603 -- trusted executable with fixed args
        [git_file, "-c", "user.name=kompend init", "-c", "user.email=kompend@local", "commit", "-m", "chore: initialise kompend wiki project"],
        cwd=target_dir,
        check=True,
        capture_output=True,
    )
