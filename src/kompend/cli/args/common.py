"""Common CLI arguments shared across multiple commands."""

from __future__ import annotations

from typing import Annotated

import typer

from kompend.core.config.keys import EnvVar

# Application name used for environment variable prefixes.
APP_NAME_UPPERCASE = "KOMPEND"

# ── Defaults ────────────────────────────────────────────────────────────

DEFAULT_LLM_FALLBACK_MAX_RETRIES: int = 2
DEFAULT_START_AT_TIER: str = "index"


# ── Global / cross-command options ──────────────────────────────────────

WikiDir = Annotated[
    str | None,
    typer.Option(
        "--wiki-dir",
        "-w",
        help="Project root directory. All other paths are resolved relative to it. Default: current working directory.",
        envvar=EnvVar.WIKI_DIR,
        rich_help_panel="Project",
    ),
]

Recursive = Annotated[
    bool,
    typer.Option(
        "--recursive/--no-recursive",
        "-r/-R",
        help="Search source directories recursively.",
        envvar=EnvVar.RECURSIVE,
    ),
]

ConfigPath = Annotated[
    str | None,
    typer.Option(
        "--config",
        "-c",
        help=("Path to configuration file. Default: auto-discover './kompend.toml' (walking up) then ~/.config/kompend/config.toml."),
        envvar=EnvVar.CONFIG,
        rich_help_panel="Project",
    ),
]

Verbose = Annotated[
    bool,
    typer.Option(
        "--verbose",
        help="Show detailed output.",
    ),
]

JsonOutput = Annotated[
    bool,
    typer.Option(
        "--json",
        help="Output in JSON format for machine readability.",
    ),
]

Progress = Annotated[
    bool | None,
    typer.Option(
        "--progress/--no-progress",
        help="Show rich progress bars (auto-detected for TTY).",
        rich_help_panel="Pipeline",
    ),
]

CliSet = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        help=(
            "Override a config value for this run only (repeatable). "
            "Format: --set compile.verify=false. Values are coerced to the "
            "key's type. Does NOT persist — use `kompend config set` for a "
            "permanent change."
        ),
        rich_help_panel="Project",
    ),
]
