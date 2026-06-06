#!/usr/bin/env python3
"""KNOX Benchmark Runner.

Runs all quality benchmarks and prints markdown-formatted results.
Can optionally save results to a file.

Usage:
    uv run python scripts/run_benchmarks.py
    uv run python scripts/run_benchmarks.py --output results.md
    uv run python scripts/run_benchmarks.py --verbose
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_benchmarks(
    output: str | None = None,
    verbose: bool = False,
) -> int:
    """Run all KNOX benchmarks and print formatted results.

    Args:
        output: Optional file path to save markdown results.
        verbose: Show full pytest output.

    Returns:
        Exit code (0 = all passed, 1 = some failed).
    """
    project_root = Path(__file__).resolve().parent.parent

    print("# KNOX Quality Benchmarks\n")
    print(f"_Run: {datetime.now().isoformat()}_\n")

    # Run chunk quality benchmarks
    print("## Chunk Quality Benchmarks\n")
    chunk_result = subprocess.run(
        [
            "uv", "run", "pytest",
            "tests/benchmarks/test_chunk_quality.py",
            "-v",
            "--tb=short",
            "-s",  # show print output
            "--no-header",
        ],
        capture_output=True,
        text=True,
        cwd=str(project_root),
    )
    if verbose or chunk_result.returncode != 0:
        print(chunk_result.stdout)
        if chunk_result.stderr:
            print(f"```\n{chunk_result.stderr}\n```")
    else:
        _print_filtered_output(chunk_result.stdout)

    # Run parser benchmarks (will skip if no PDF found)
    print("\n## Parser Quality Benchmarks\n")
    parser_result = subprocess.run(
        [
            "uv", "run", "pytest",
            "tests/benchmarks/test_parse_quality.py",
            "-v",
            "--tb=short",
            "-s",
            "--no-header",
        ],
        capture_output=True,
        text=True,
        cwd=str(project_root),
    )
    if verbose or parser_result.returncode != 0:
        print(parser_result.stdout)
        if parser_result.stderr:
            print(f"```\n{parser_result.stderr}\n```")
    else:
        _print_filtered_output(parser_result.stdout)

    # Collect summary
    print("\n## Summary\n")
    print(f"| Benchmarks | Result |")
    print(f"|---|---|")
    for name, result in [
        ("Chunk Quality", chunk_result),
        ("Parser Quality", parser_result),
    ]:
        status = "✅ PASS" if result.returncode == 0 else "❌ FAIL"
        total = _extract_total(result.stdout)
        skipped = _extract_skipped(result.stdout)
        details = f" ({total} tests"
        if skipped:
            details += f", {skipped} skipped"
        details += ")"
        print(f"| {name} | {status}{details} |")

    # Optionally save
    if output:
        _save_report(output, chunk_result, parser_result)
        print(f"\n📄 Report saved to `{output}`\n")

    return chunk_result.returncode or parser_result.returncode


def _print_filtered_output(text: str) -> None:
    """Print only the relevant lines (test names + print output + summary)."""
    for line in text.splitlines():
        if line.startswith("  ") or line.startswith("PASSED") or line.startswith("FAILED") or line.startswith("SKIPPED"):
            print(line)
        elif "benchmark" in line.lower() or "chunk" in line.lower() or "token" in line.lower():
            print(line)
        elif "passed" in line and "failed" in line and "=" in line:
            print(line)


def _extract_total(output: str) -> str:
    """Extract total test count from pytest summary line."""
    for line in output.splitlines():
        if "passed" in line and "=" in line:
            parts = line.split()
            for p in parts:
                if p.isdigit():
                    return p
    return "?"


def _extract_skipped(output: str) -> str:
    """Extract skipped count from pytest summary line."""
    for line in output.splitlines():
        if "skipped" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "skipped" and i > 0 and parts[i - 1].isdigit():
                    return parts[i - 1]
    return ""


def _save_report(output: str, chunk_result, parser_result) -> None:
    """Write full report to file."""
    path = Path(output)
    dt = datetime.now().isoformat()

    lines = [
        f"# KNOX Quality Benchmarks Report",
        f"",
        f"_Run: {dt}_",
        f"",
        "## Chunk Quality",
        f"```",
        chunk_result.stdout if chunk_result.returncode == 0 else chunk_result.stdout + "\nSTDERR:\n" + chunk_result.stderr,
        f"```",
        f"",
        "## Parser Quality",
        f"```",
        parser_result.stdout if parser_result.returncode == 0 else parser_result.stdout + "\nSTDERR:\n" + parser_result.stderr,
        f"```",
        f"",
        "## Exit Codes",
        f"",
        f"- Chunk Quality: {chunk_result.returncode}",
        f"- Parser Quality: {parser_result.returncode}",
        f"",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run KNOX quality benchmarks")
    parser.add_argument(
        "--output", "-o",
        help="Save report to file (markdown)",
        default=None,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full pytest output",
    )
    args = parser.parse_args()
    return run_benchmarks(output=args.output, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())