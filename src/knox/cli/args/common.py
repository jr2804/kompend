"""Common CLI arguments shared across multiple commands."""

from __future__ import annotations

from typing import Annotated

import typer

from knox.core.config.keys import EnvVar

# Application name used for environment variable prefixes.
APP_NAME_UPPERCASE = "KNOX"

# ── Defaults ────────────────────────────────────────────────────────────

DEFAULT_LLM_FALLBACK_MAX_RETRIES: int = 2
DEFAULT_START_AT_TIER: str = "index"


# ── Global / cross-command options ──────────────────────────────────────

WikiDir = Annotated[
    str | None,
    typer.Option(
        "--wiki-dir",
        "-w",
        help="Project root directory. All other paths are resolved relative to it. Default: current working directory. Env: KNOX_WIKI_DIR.",
        envvar=EnvVar.WIKI_DIR,
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
        help=(
            f"Path to configuration file. Default: auto-discover './knox.toml' (walking up) then ~/.config/knox/config.toml. Env: {APP_NAME_UPPERCASE}_CONFIG."
        ),
        envvar=EnvVar.CONFIG,
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

CliSet = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        help=(
            "Override a config value for this run only (repeatable). "
            "Format: --set compile.verify=false. Values are coerced to the "
            "key's type. Does NOT persist — use `knox config set` for a "
            "permanent change."
        ),
        rich_help_panel="Overrides",
    ),
]

Parser = Annotated[
    str | None,
    typer.Option(
        "--parser",
        help=(
            "Ingestion backend name, or a comma-separated priority list "
            "(e.g. 'liteparse,docling'). The first backend that supports a "
            "file's format wins; list multiple to keep coverage of formats "
            "the primary lacks. See `knox doctor` for available backends. "
            f"Env: {APP_NAME_UPPERCASE}_INGEST_PARSER."
        ),
        envvar=EnvVar.INGEST_PARSER,
    ),
]
