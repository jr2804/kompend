"""``kompend init`` — scaffold a new wiki project."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import jinja2
import typer

from kompend.cli.app import app
from kompend.cli.args.init import InitBare, InitConfig, InitForce, InitGit, InitKeepCache, InitTarget, InitWikiDir
from kompend.logging import get_console, get_err_console, get_logger
from kompend.scaffolding import TEMPLATE_NEXT_STEPS, ScaffoldError, find_existing_project, scaffold_project

logger = get_logger(__name__)


@app.command(name="init", help="Scaffold a new kompend wiki project.")
def init(
    target: InitTarget = ".",
    bare: InitBare = False,
    force: InitForce = False,
    git: InitGit = False,
    keep_cache: InitKeepCache = False,
    seed_config: InitConfig = None,
    wiki_dir: InitWikiDir = None,
) -> None:
    """Create a new wiki project in TARGET (default: current directory)."""
    if wiki_dir is not None:
        if target != "." and Path(target) != Path(wiki_dir):
            get_err_console().print("error: pass the project directory either as TARGET or via --wiki-dir, not both.")
            raise typer.Exit(code=1)
        target = wiki_dir
    target_dir = Path(target)

    if not force:
        existing = find_existing_project(target_dir)
        if existing is not None:
            get_err_console().print(
                f"warning: existing kompend project detected at {existing}.\n  Use --force to overwrite (this wipes the current vault).",
            )
            raise typer.Exit(code=0)

    try:
        result = scaffold_project(target_dir, bare=bare, force=force, git=git, keep_cache=keep_cache, config_file=seed_config)
    except ScaffoldError as exc:
        get_err_console().print(f"error: {exc}")
        raise typer.Exit(code=1) from exc

    get_console().print(f"Created wiki project at {result.target_dir}")
    for directory in result.created_dirs:
        get_console().print(f"  + {directory.relative_to(result.target_dir)}/")
    for file in result.copied_files:
        get_console().print(f"  + {file.relative_to(result.target_dir)}")
    if result.skipped_files:
        get_console().print(f"  - skipped (--bare): {', '.join(result.skipped_files)}")
    if result.clobbered:
        get_err_console().print("--force overwrote the following tracked files (use git checkout to restore):")
        for clobbered in result.clobbered:
            get_err_console().print(f"    - {clobbered.relative_to(result.target_dir)}")

    if result.git_initialised:
        get_console().print("  + git init: initial commit created")
    get_console().print("")
    get_console().print(_load_next_steps(result.sources_dir or "sources"), end="")


def _load_next_steps(sources_dir: str) -> str:
    """Load and render the next-steps template with the configured sources_dir."""
    content = resources.files("kompend.scaffolding.templates").joinpath(TEMPLATE_NEXT_STEPS).read_text(encoding="utf-8")
    return jinja2.Template(content, keep_trailing_newline=True).render(sources_dir=sources_dir)
