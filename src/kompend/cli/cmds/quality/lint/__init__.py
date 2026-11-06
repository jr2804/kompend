"""The ``kompend lint`` command group.

Sub-commands
------------

* ``kompend lint check``     — deterministic structural rules and their fixes.
* ``kompend lint semantic``  — LLM rules: factual consistency, clarity, coverage, contradictions.
* ``kompend lint all``       — every step in one weighted run.
* ``kompend lint format``    — rumdl over the generated markdown.
* ``kompend lint history``   — past runs and score trends.
* ``kompend lint show``      — the full report of one run.
* ``kompend lint prune``     — trim the history log.
* ``kompend lint gaps``      — knowledge gaps and source recommendations.
* ``kompend lint remediate`` — generate the missing pages.

Registration is imperative and lives here so each module below stays a plain
module of functions: a sub-command's name and its short help are stated once,
in this file, and the docstring of the function it registers stays the long
help.
"""

from __future__ import annotations

from cyclopts import App

from kompend.cli.cmds.quality.lint.formatting import lint_format
from kompend.cli.cmds.quality.lint.gaps import lint_gaps
from kompend.cli.cmds.quality.lint.history import lint_history, lint_prune, lint_show
from kompend.cli.cmds.quality.lint.remediate import lint_remediate
from kompend.cli.cmds.quality.lint.run import lint_all, lint_check, lint_semantic

__all__ = ["lint_app"]

lint_app = App(name="lint", help="Lint a compiled wiki for quality issues and query lint history.")

lint_app.command(lint_check, name="check", help="Run deterministic structural lint rules (no LLM).")
lint_app.command(
    lint_semantic,
    name="semantic",
    help="Run LLM-powered lint rules (factual-consistency, clarity, coverage-gap, contradictions).",
)
lint_app.command(lint_all, name="all", help="Run the full lint pipeline: check, format, semantic, gaps, remediate.")
lint_app.command(lint_history, name="history", help="Show past lint runs and score trends.")
lint_app.command(lint_show, name="show", help="Print the detailed report of a lint run (default: the newest).")
lint_app.command(lint_prune, name="prune", help="Remove old entries from the lint history.")
lint_app.command(lint_gaps, name="gaps", help="Analyze lint results for knowledge gaps and source recommendations.")
lint_app.command(lint_remediate, name="remediate", help="LLM-generate missing pages for lint findings.")
lint_app.command(lint_format, name="format", help="Format generated wiki markdown with rumdl fmt.")
