"""``knox doctor`` — validate configuration and connectivity."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from knox.preprocessing.parsers.base import ParserRouter

logger = logging.getLogger(__name__)

# ── Check results ───────────────────────────────────────────────────────


class CheckResult:
    """Result of a single doctor check."""

    def __init__(self, name: str, passed: bool, message: str = "") -> None:
        self.name = name
        self.passed = passed
        self.message = message

    def to_dict(self) -> dict[str, str | bool]:
        return {"name": self.name, "passed": self.passed, "message": self.message}


# ── Checks ──────────────────────────────────────────────────────────────


def _check_sources_dir(sources_dir: Path | None) -> CheckResult:
    name = "sources-directory"
    if sources_dir is None:
        return CheckResult(name, True, "No sources dir configured — skipped")
    if not sources_dir.exists():
        return CheckResult(name, False, f"Sources dir does not exist: {sources_dir}")
    if not sources_dir.is_dir():
        return CheckResult(name, False, f"Sources path is not a directory: {sources_dir}")
    return CheckResult(name, True, f"Sources dir OK: {sources_dir}")


def _check_output_dir(output_dir: Path | None) -> CheckResult:
    name = "output-directory"
    if output_dir is None:
        return CheckResult(name, True, "No output dir configured — skipped")
    if not output_dir.exists():
        return CheckResult(name, False, f"Output dir does not exist: {output_dir}")
    if not os.access(str(output_dir), os.W_OK):
        return CheckResult(name, False, f"Output dir not writable: {output_dir}")
    return CheckResult(name, True, f"Output dir OK: {output_dir}")


def _check_parsers() -> CheckResult:
    name = "parsers"
    router = ParserRouter()
    try:
        from knox.preprocessing.parsers.docling_parser import DoclingParser  # noqa: PLC0415
        router.register(DoclingParser())
    except Exception as exc:
        return CheckResult(name, False, f"DoclingParser init failed: {exc}")

    try:
        from knox.preprocessing.parsers.pdfplumber_parser import PdfPlumberParser  # noqa: PLC0415
        router.register(PdfPlumberParser())
    except Exception:
        pass  # optional parser

    try:
        from knox.preprocessing.parsers.pandoc_parser import PandocParser  # noqa: PLC0415
        router.register(PandocParser())
    except Exception:
        pass  # optional parser

    return CheckResult(name, True, f"{len(router._parsers)} parser(s) registered")


def _check_env_vars(*vars_to_check: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    for var_name in vars_to_check:
        if os.environ.get(var_name):
            results.append(CheckResult(f"env:{var_name}", True, f"{var_name} is set"))
        else:
            results.append(CheckResult(f"env:{var_name}", False, f"{var_name} is not set"))
    return results


def _check_llm_connectivity(verbose: bool = False) -> CheckResult:
    """Quick connectivity check — pings the configured LLM provider."""
    name = "llm-connectivity"
    model = os.environ.get("KNOX_LLM_MODEL", "llama3.2")
    provider = os.environ.get("KNOX_LLM_PROVIDER", "ollama")

    try:
        import litellm  # noqa: PLC0415
    except ImportError:
        return CheckResult(name, False, "litellm not installed")

    try:
        response = litellm.completion(
            model=f"{provider}/{model}",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
            timeout=5,
        )
        model_used = getattr(response, "model", "unknown")
        return CheckResult(name, True, f"LLM OK ({provider}/{model_used})")
    except Exception as exc:
        msg = f"LLM ping failed: {exc}"
        if verbose:
            msg += f" (model={provider}/{model})"
        return CheckResult(name, False, msg)


def _check_git_repo() -> CheckResult:
    """Check if we are inside a git repository."""
    name = "git-repo"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return CheckResult(name, True, "Inside a git repository")
        return CheckResult(name, True, "Not a git repository — OK")
    except FileNotFoundError:
        return CheckResult(name, True, "Git not installed — OK")
    except subprocess.TimeoutExpired:
        return CheckResult(name, True, "Git check timed out — OK")


def _check_disk_space(sources_dir: Path | None = None) -> CheckResult:
    """Check available disk space — warn if < 500MB."""
    name = "disk-space"
    check_path = sources_dir if sources_dir and sources_dir.exists() else Path.cwd()
    try:
        stat = os.statvfs(str(check_path))
        free_bytes = stat.f_frsize * stat.f_bavail
        free_mb = free_bytes / (1024 * 1024)
        if free_mb < 500:
            return CheckResult(name, False, f"Low disk space: {free_mb:.0f} MB free (< 500 MB)")
        return CheckResult(name, True, f"{free_mb:.0f} MB free")
    except Exception:
        return CheckResult(name, True, "Could not check disk space — skipped")


# ── Doctor runner ───────────────────────────────────────────────────────


def run_doctor(
    sources_dir: Path | None = None,
    output_dir: Path | None = None,
    verbose: bool = False,
) -> list[CheckResult]:
    """Run all doctor checks and return results."""
    results: list[CheckResult] = []

    results.append(_check_sources_dir(sources_dir))
    results.append(_check_output_dir(output_dir))
    results.append(_check_parsers())
    results.append(_check_git_repo())
    results.append(_check_disk_space(sources_dir))
    results.extend(_check_env_vars("KNOX_LLM_MODEL", "KNOX_LLM_PROVIDER", "OPENAI_API_KEY"))

    if verbose:
        results.append(_check_llm_connectivity(verbose=True))
    else:
        results.append(_check_llm_connectivity())

    return results


def print_doctor_results(results: list[CheckResult], json_output: bool = False) -> int:
    """Print doctor results. Returns exit code (0 = all passed)."""
    passed_count = sum(1 for r in results if r.passed)
    failed_count = sum(1 for r in results if not r.passed)

    if json_output:
        import json as j  # noqa: PLC0415
        data = {
            "passed": passed_count,
            "failed": failed_count,
            "total": len(results),
            "checks": [r.to_dict() for r in results],
        }
        j.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for r in results:
            icon = "[OK]" if r.passed else "[!!]"
            print(f"  {icon} {r.name}: {r.message}")

        total = len(results)
        print(f"\n  {passed_count}/{total} checks passed", end="")
        if failed_count:
            print(f", {failed_count} failed", end="")
        print()

    return 1 if failed_count else 0
