"""Common CLI arguments shared across multiple commands."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Any, Literal

from cyclopts import Parameter

from kompend.core.keys import EnvVar

# Application name used for environment variable prefixes.
APP_NAME_UPPERCASE = "kompend"

# ── Defaults ────────────────────────────────────────────────────────────

DEFAULT_START_AT_TIER: str = "index"

# ── Help panel titles ───────────────────────────────────────────────────

PROJECT_GROUP = "Project"
SOURCES_GROUP = "Sources"
PIPELINE_GROUP = "Pipeline"
OUTPUT_GROUP = "Output"
LLM_GROUP = "LLM"


#: Cyclopts' own parameter type for the shells in :class:`CompletionShell`.
#: It cannot be derived from the enum (``Literal`` is static), so the adapter
#: in ``cli/app.py`` casts the member value into it.
ShellName = Literal["zsh", "bash", "fish", "powershell"]


# ── Global / cross-command options ──────────────────────────────────────

WikiDir = Annotated[
    str | None,
    Parameter(
        name=("--wiki-dir", "-w"),
        help="Project root directory. All other paths are resolved relative to it. Default: current working directory.",
        env_var=EnvVar.WIKI_DIR,
        group=PROJECT_GROUP,
    ),
]

RecursiveScan = Annotated[
    bool | None,
    Parameter(
        name="--recursive",
        help="Scan sources/ subdirectories for source files. Default: paths.recursive config value (on).",
        group=SOURCES_GROUP,
    ),
]

ConfigPath = Annotated[
    str | None,
    Parameter(
        name=("--config", "-c"),
        help="Path to configuration file. Default: auto-discover './kompend.toml' (walking up) then ~/.config/kompend/config.toml.",
        env_var=EnvVar.CONFIG,
        group=PROJECT_GROUP,
    ),
]

Verbose = Annotated[
    bool,
    Parameter(
        name="--verbose",
        help="Show full tracebacks for errors (fatal crashes and recoverable parse failures). Env: KOMP_VERBOSE.",
        env_var=EnvVar.VERBOSE,
        negative="",
    ),
]

JsonOutput = Annotated[
    bool,
    Parameter(
        name="--json",
        help="Output in JSON format for machine readability.",
        negative="",
    ),
]

Progress = Annotated[
    bool | None,
    Parameter(
        name="--progress",
        help="Show rich progress bars (auto-detected for TTY).",
        group=PIPELINE_GROUP,
    ),
]

NoUserTracking = Annotated[
    bool,
    Parameter(
        name="--no-user-tracking",
        help="Do not record the OS user (ingested_by / memory_by) in metadata for this invocation. Env: KOMP_USER_TRACKING=false.",
        negative="",
    ),
]

CliSet = Annotated[
    list[str] | None,
    Parameter(
        negative="",
        name="--set",
        help=(
            "Override a config value for this run only (repeatable). "
            "Format: --set compile.verify=false. Values are coerced to the "
            "key's type. Does NOT persist — use `kompend config set` for a "
            "permanent change."
        ),
        group=PROJECT_GROUP,
    ),
]


# ── Vocabularies ────────────────────────────────────────────────────────


class CompletionShell(StrEnum):
    """Shells the completion command can print a script for.

    Mirrors the vocabulary Cyclopts' ``generate_completion`` accepts; the
    enum exists so the CLI surface names each shell once instead of repeating
    a ``Literal`` in the parameter annotation.
    """

    ZSH = "zsh"
    BASH = "bash"
    FISH = "fish"
    POWERSHELL = "powershell"


def shell_name(shell: CompletionShell | None) -> ShellName | None:
    """Narrow a shell member to the literal Cyclopts accepts.

    The two declarations cannot be merged (one is an enum, the other a static
    literal), so one function owns the narrowing instead of every call site.
    """
    return shell.value if shell is not None else None


def in_range(low: int, high: int | None = None) -> Callable[[Any, Any], None]:
    """Return a validator rejecting values outside ``low``..``high``.

    Cyclopts has no declarative bound on :class:`Parameter`, so numeric limits
    are expressed as validators; raising :class:`ValueError` turns into a usage
    error (exit code 2).
    """
    expected = f"between {low} and {high}" if high is not None else f"{low} or greater"

    def _validate(type_: Any, value: Any) -> None:  # noqa: ANN401 - Cyclopts signature
        if value is None:
            return
        if value < low or (high is not None and value > high):
            raise ValueError(f"must be {expected}, got {value}")

    return _validate
