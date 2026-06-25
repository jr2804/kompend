"""Common CLI arguments shared across multiple commands."""

from __future__ import annotations

from typing import Annotated

import typer

from knox.core.pipeline_defs import Compiler
from knox.core.pipeline_defs import ImageMode as ImageModeEnum
from knox.core.pipeline_defs import OutputFormat as OutputFormatEnum

# Application name used for environment variable prefixes.
APP_NAME_UPPERCASE = "KNOX"

# ── Defaults ────────────────────────────────────────────────────────────

DEFAULT_COMPILE_FORMAT: OutputFormatEnum = OutputFormatEnum.MARKDOWN
DEFAULT_IMAGE_MODE: ImageModeEnum = ImageModeEnum.FILE
DEFAULT_PROMPT_BUDGET: int = 200_000
DEFAULT_LLM_FALLBACK_MAX_RETRIES: int = 2
DEFAULT_COMPILER: Compiler = Compiler.GRAPH
DEFAULT_START_AT_TIER: str = "index"


# ── Global / cross-command options ──────────────────────────────────────

WikiDir = Annotated[
    str | None,
    typer.Option(
        "--wiki-dir",
        "-w",
        help="Project root directory. All other paths are resolved relative to it. Default: current working directory. Env: KNOX_WIKI_DIR.",
        envvar=f"{APP_NAME_UPPERCASE}_WIKI_DIR",
    ),
]

Recursive = Annotated[
    bool,
    typer.Option(
        "--recursive/--no-recursive",
        "-r/-R",
        help="Search source directories recursively.",
        envvar=f"{APP_NAME_UPPERCASE}_RECURSIVE",
    ),
]

ConfigPath = Annotated[
    str | None,
    typer.Option(
        "--config",
        "-c",
        help=(
            f"Path to configuration file. Default: auto-discover './knox.toml' (walking up) then ~/.config/knox/config.toml. Env: {APP_NAME_UPPERCASE}_CONFIG."
        ),
        envvar=f"{APP_NAME_UPPERCASE}_CONFIG",
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
    ),
]
