"""Host and configuration diagnostics behind ``kompend doctor``.

The checks live here, not in the CLI: the command layer renders them and the
MCP server exposes the same results as a tool, so neither may own the logic.
:mod:`kompend.diagnostics.runner` decides which checks run and in which order;
every other module answers one family of questions.
"""

from __future__ import annotations

from kompend.diagnostics.base import CheckResult
from kompend.diagnostics.runner import run_doctor

__all__ = ["CheckResult", "run_doctor"]
