"""``knox doctor`` — validate configuration and connectivity."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import litellm

from knox.llm.client import LLMClient
from knox.llm.roles import LLMRole
from knox.logging import get_console, get_logger
from knox.preprocessing.parsers._registry import build_default_router

logger = get_logger(__name__)

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
    """Check parser availability via the shared registry factory."""
    name = "parsers"
    router = build_default_router()
    return CheckResult(name, True, f"{len(router._parsers)} parser(s) registered")


def _ping_model(model: str | None, label: str) -> CheckResult:
    """Ping a single completion-type model (LLM or VLM)."""
    name = f"connectivity-{label}"
    if not model:
        return CheckResult(name, True, f"No {label} model configured — skipped")

    try:
        client = LLMClient(default_model=model)
        result = client.try_complete(
            [{"role": "user", "content": "Respond with exactly one word: ok"}],
            role=LLMRole.DOCTOR,
            # Reasoning models (e.g. deepseek-v4-flash) emit reasoning_content
            # before the answer; a tiny budget is consumed by thinking and the
            # visible content stays empty -> false negative on connectivity.
            # 128 gives reasoning headroom while staying below litellm's
            # repetition-loop threshold (some vision models spam on a text ping).
            max_tokens=128,
        )
        if result is not None:
            model_used = result.diagnostics.model_used
            return CheckResult(name, True, f"{label} OK ({model_used})")
        return CheckResult(name, False, f"{label} returned no response")
    except Exception as exc:
        return CheckResult(name, False, f"{label} failed: {exc}")


def _ping_embed_model(model: str | None) -> CheckResult:
    """Ping an embedding model via ``litellm.embedding()``."""
    name = "connectivity-embed"
    if not model:
        return CheckResult(name, True, "No embed model configured — skipped")

    # The ``litellm:`` prefix is a KNOX convention — strip it before
    # passing to litellm, otherwise litellm can't parse the provider.
    effective = model.removeprefix("litellm:")

    try:
        response = litellm.embedding(model=effective, input="ping")
        dims = len(response.data[0]["embedding"]) if response.data else 0
        return CheckResult(name, True, f"embed OK ({effective}, {dims} dims)")
    except Exception as exc:
        return CheckResult(name, False, f"embed failed: {exc}")


def _check_git_repo() -> CheckResult:
    """Check if we are inside a git repository."""
    name = "git-repo"
    git_bin = shutil.which("git")
    if git_bin is None:
        return CheckResult(name, True, "Git not installed — OK")
    try:
        result = subprocess.run(  # noqa: S603
            [git_bin, "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return CheckResult(name, True, "Inside a git repository")
        return CheckResult(name, True, "Not a git repository — OK")
    except subprocess.TimeoutExpired:
        return CheckResult(name, True, "Git check timed out — OK")


def _check_disk_space(sources_dir: Path | None = None) -> CheckResult:
    """Check available disk space — warn if < 500 MB."""
    name = "disk-space"
    check_dir = sources_dir if sources_dir and sources_dir.exists() else Path.cwd()
    try:
        usage = shutil.disk_usage(str(check_dir))
        free_mb = usage.free / (1024 * 1024)
        if free_mb < 500:
            return CheckResult(name, False, f"Low disk space: {free_mb:.0f} MB free (< 500 MB)")
        return CheckResult(name, True, f"{free_mb:.0f} MB free")
    except Exception:
        return CheckResult(name, True, "Could not check disk space — skipped")


# ── Doctor runner ───────────────────────────────────────────────────────


def run_doctor(
    sources_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    default_model: str | None = None,
    vlm_model: str | None = None,
    embed_model: str | None = None,
) -> list[CheckResult]:
    """Run all doctor checks and return results.

    Parameters
    ----------
    sources_dir:
        Sources directory to validate, or ``None`` to skip.
    output_dir:
        Output directory to validate, or ``None`` to skip.
    default_model:
        LLM model name (litellm format) for the connectivity ping.
    vlm_model:
        VLM model name for the connectivity ping.
    embed_model:
        Embedding model name for the connectivity ping.
    """
    results: list[CheckResult] = []

    results.append(_check_sources_dir(sources_dir))
    results.append(_check_output_dir(output_dir))
    results.append(_check_parsers())
    results.append(_check_git_repo())
    results.append(_check_disk_space(sources_dir))
    results.append(_ping_model(default_model, "llm"))
    results.append(_ping_model(vlm_model, "vlm"))
    results.append(_ping_embed_model(embed_model))

    return results


def print_doctor_results(results: list[CheckResult], json_output: bool = False) -> int:
    """Print doctor results. Returns exit code (0 = all passed)."""
    passed_count = sum(1 for r in results if r.passed)
    failed_count = sum(1 for r in results if not r.passed)

    if json_output:
        data = {
            "passed": passed_count,
            "failed": failed_count,
            "total": len(results),
            "checks": [r.to_dict() for r in results],
        }
        json.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for r in results:
            icon = "[OK]" if r.passed else "[!!]"
            get_console().print(f"  {icon} {r.name}: {r.message}", markup=False)

        total = len(results)
        get_console().print(f"\n  {passed_count}/{total} checks passed", end="")
        if failed_count:
            get_console().print(f", {failed_count} failed", end="")
        get_console().print("")

    return 1 if failed_count else 0
