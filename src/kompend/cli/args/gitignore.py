"""CLI argument types for the ``gitignore`` command."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

GitignoreCheck = Annotated[
    bool,
    Parameter(
        name="--check",
        help="Exit 1 when the vault's .gitignore lacks the managed kompend block or carries an outdated one.",
        negative="",
    ),
]

GitignorePrint = Annotated[
    bool,
    Parameter(
        name="--print",
        help="Print the rendered .gitignore instead of writing it.",
        negative="",
    ),
]
