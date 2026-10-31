"""CLI arguments for the `kompend memory` command group."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from kompend.cli.args.common import in_range

MemoryQuery = Annotated[
    str,
    Parameter(help="The query string to search for or add to memory."),
]

MemoryAnswer = Annotated[
    str,
    Parameter(help="The answer or response to store in memory."),
]

MemoryFeedback = Annotated[
    str | None,
    Parameter(
        name=("--feedback", "-f"),
        help="Optional user feedback or correction for this entry.",
    ),
]

MemoryTopK = Annotated[
    int,
    Parameter(
        name=("--top-k", "-k"),
        help="Number of top results to return.",
        validator=in_range(1, 50),
    ),
]

MemoryLimit = Annotated[
    int,
    Parameter(
        name=("--limit", "-l"),
        help="Number of recent entries to synthesize.",
        validator=in_range(1, 100),
    ),
]
