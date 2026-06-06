"""CLI command implementations for My Project."""

from __future__ import annotations

import typer

from knox.cli import args
from knox.cli.app import app

# Application name for environment variables
APP_NAME_UPPERCASE = "KNOX"

# Default command (runs when no command provided)

@app.command()
def default() -> None:
    """Default command showing welcome message."""
    typer.echo(f"Welcome to My Project!")
    typer.echo("Use --help to see available commands.")

# Greet command

@app.command()
def greet(
    name: args.NameArg,
    input_file: args.InputFileArg = None,
    output_file: args.OutputFile = None,
    cache: args.Cache = True,
) -> None:
    """Greet a person by name."""
    if input_file:
        # Read names from file
        names = [line.strip() for line in input_file if line.strip()]
        for n in names:
            typer.echo(f"Hello, {n}!")
    else:
        typer.echo(f"Hello, {name}!")

    # Show option values (for demonstration)
    if output_file:
        typer.echo(f"Output file: {output_file}")
    typer.echo(f"Cache: {'enabled' if cache else 'disabled'}")

# Add command

@app.command()
def add(
    number1: args.NumberArg1,
    number2: args.NumberArg2,
    output_file: args.OutputFile = None,
) -> None:
    """Add two numbers together."""
    result = number1 + number2
    typer.echo(f"The sum of {number1} and {number2} is {result}")

    if output_file:
        with open(output_file, "w") as f:
            f.write(str(result))
        typer.echo(f"Result written to {output_file}")
