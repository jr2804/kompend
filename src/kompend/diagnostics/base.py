"""The result type every doctor check returns.

A check answers one question about the host or the configuration and never
raises: an unreachable provider, a missing wheel or a stale catalogue entry is
a fact the doctor prints, not an exception that aborts the run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of a single doctor check.

    Attributes:
        name: Stable kebab-case identifier — the JSON output and the MCP tool
            expose it, so it is part of the contract.
        passed: Whether the check found the state acceptable.
        message: Human-readable detail, shown verbatim.
        warning: A passing check whose state is worth reading anyway (a
            degraded fallback, a skipped capability).
    """

    name: str
    passed: bool = True
    message: str = ""
    warning: bool = False

    def to_dict(self) -> dict[str, str | bool]:
        """Return the JSON shape ``kompend doctor --json`` and the MCP tool emit."""
        return {"name": self.name, "passed": self.passed, "message": self.message, "warning": self.warning}
