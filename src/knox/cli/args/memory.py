"""CLI arguments for the `knox memory` command group."""

from __future__ import annotations

from typing import Annotated

import typer

MemoryQuery = Annotated[
    str,
    typer.Argument(help="The query string to search for or add to memory."),
]

MemoryAnswer = Annotated[
    str,
    typer.Argument(help="The answer or response to store in memory."),
]

MemoryFeedback = Annotated[
    str | None,
    typer.Option("--feedback", "-f", help="Optional user feedback or correction for this entry."),
]

MemoryTopK = Annotated[
    int,
    typer.Option("--top-k", "-k", help="Number of top results to return.", min=1, max=50),
]

MemoryLimit = Annotated[
    int,
    typer.Option("--limit", "-l", help="Number of recent entries to synthesize.", min=1, max=100),
]
