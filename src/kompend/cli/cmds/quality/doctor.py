"""``kompend doctor`` output — the text and JSON rendering of the checks.

The checks themselves live in :mod:`kompend.diagnostics`; this module only
decides how their results reach the user and what exit code they imply.
"""

from __future__ import annotations

import json
import sys

from kompend.diagnostics.base import CheckResult
from kompend.logging import get_console


def print_doctor_results(results: list[CheckResult], json_output: bool = False) -> int:
    """Print doctor results. Returns exit code (0 = no failures)."""
    failed_count = sum(1 for result in results if not result.passed)
    warn_count = sum(1 for result in results if result.passed and result.warning)
    ok_count = sum(1 for result in results if result.passed and not result.warning)

    if json_output:
        data = {
            "passed": ok_count,
            "warning": warn_count,
            "failed": failed_count,
            "total": len(results),
            "checks": [result.to_dict() for result in results],
        }
        json.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        console = get_console()
        for result in results:
            if not result.passed:
                icon = "[!]"
            elif result.warning:
                icon = "[WARN]"
            else:
                icon = "[OK]"
            console.print(f"  {icon} {result.name}: {result.message}", markup=False)

        total = len(results)
        console.print(f"\n  {ok_count}/{total} checks passed", end="")
        if warn_count:
            console.print(f", {warn_count} warning(s)", end="")
        if failed_count:
            console.print(f", {failed_count} failed", end="")
        console.print("")

    return 1 if failed_count else 0
