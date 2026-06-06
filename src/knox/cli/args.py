"""CLI arguments, options, and flags for My Project."""

from __future__ import annotations

from typing import Annotated

import typer

# Application name for environment variables
APP_NAME_UPPERCASE = "KNOX"

# Global options

OutputFile = Annotated[
    str | None,
    typer.Option(
        "--output-file",
        "-o",
        help="Path to output file",
        envvar=f"{APP_NAME_UPPERCASE}_OUTPUT_FILE",
    ),
]

Cache = Annotated[
    bool,
    typer.Option(
        "--cache/--no-cache",
        help="Enable or disable caching",
        envvar=f"{APP_NAME_UPPERCASE}_CACHE",
    ),
]

# Command-specific arguments

NameArg = Annotated[
    str,
    typer.Argument(help="Name to greet"),
]

InputFileArg = Annotated[
    typer.FileText | None,
    typer.Option(
        "--input-file",
        help="File containing names to greet (one per line)",
    ),
]

NumberArg1 = Annotated[
    float,
    typer.Argument(help="First number"),
]

NumberArg2 = Annotated[
    float,
    typer.Argument(help="Second number"),
]
