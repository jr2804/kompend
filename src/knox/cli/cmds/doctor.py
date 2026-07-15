"""``knox doctor`` — validate configuration and connectivity."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import litellm

from knox.agents.doctor import DoctorAgent, DoctorDeps
from knox.core.config.keys import COMPILE_LLM_OUTPUT_RETRIES
from knox.llm.context_window import discover_context_window
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


def _check_context_window(model: str | None, label: str) -> CheckResult:
    """Discover and report the context window for a model."""
    name = f"context-window-{label}"
    if not model:
        return CheckResult(name, True, f"No {label} model configured — skipped")

    try:
        cw = discover_context_window(model)
        source_label = cw.source.value if hasattr(cw.source, "value") else str(cw.source)
        if cw.source == "fallback":
            return CheckResult(
                name,
                True,
                f"{label}: {cw.max_input} in / {cw.max_output} out tokens ({source_label} — model not found in registry)",
            )
        return CheckResult(
            name,
            True,
            f"{label}: {cw.max_input} in / {cw.max_output} out tokens (source: {source_label})",
        )
    except Exception as exc:
        return CheckResult(name, False, f"{label} context window discovery failed: {exc}")


def _check_budget_consistency(model: str | None, cfg: dict[str, Any] | None) -> CheckResult:
    """Check that budget config values don't exceed the model's context window."""
    name = "budget-consistency"
    if not model or cfg is None:
        return CheckResult(name, True, "No model or config — skipped")

    try:
        cw = discover_context_window(model)
    except Exception as exc:
        return CheckResult(name, False, f"Cannot discover context window: {exc}")

    issues: list[str] = []

    # Check gen_output_max vs context window
    gen_max = _cfg_get_int(cfg, "compile.budget.gen_output_max", 32768)
    if gen_max > cw.max_output:
        issues.append(f"gen_output_max ({gen_max}) > context window output ({cw.max_output})")

    # Check extract_output_max vs context window
    extract_max = _cfg_get_int(cfg, "compile.budget.extract_output_max", 16384)
    if extract_max > cw.max_output:
        issues.append(f"extract_output_max ({extract_max}) > context window output ({cw.max_output})")

    # Check that input + output fits in context window
    sys_tokens = 128  # approximate system prompt overhead
    input_budget = cw.max_input - sys_tokens - gen_max
    if input_budget < 512:
        issues.append(
            f"Input budget ({input_budget} tokens) too small after reserving "
            f"{sys_tokens} system + {gen_max} output tokens in {cw.max_input} context window. "
            f"Reduce gen_output_max or increase context_window_cap."
        )

    if not issues:
        return CheckResult(
            name,
            True,
            f"Budgets OK: {cw.max_input} ctx, {gen_max} gen_out, {extract_max} extract_out, ~{input_budget} input",
        )
    return CheckResult(name, False, "; ".join(issues))


def _cfg_get_int(cfg: dict[str, Any], key: str, default: int) -> int:
    """Read an int from a dotted config key."""
    parts = key.split(".")
    val: Any = cfg
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p, {})
        else:
            return default
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        try:
            return int(val)
        except (ValueError, TypeError):
            return default
    return default


def _enable_litellm_debug() -> None:
    """Enable litellm debug output to stderr."""
    litellm._turn_on_debug()
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    for logger_name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            logger.addHandler(handler)


def _ping_model(model: str | None, label: str, verbose: bool = False, timeout: float = 60.0, retries: int | None = None) -> CheckResult:
    """Ping a single completion-type model (LLM or VLM) via PydanticAI DoctorAgent."""
    name = f"connectivity-{label}"
    if not model:
        return CheckResult(name, True, f"No {label} model configured — skipped")

    try:
        if verbose:
            _enable_litellm_debug()
        agent = DoctorAgent(
            litellm_model=model,
            timeout=timeout,
            max_tokens=128,
            retries=retries or 2,
        )
        result = agent.run(DoctorDeps(ping="Respond with exactly one word: ok"))
        if result and result.strip():
            return CheckResult(name, True, f"{label} OK ({model})")
        return CheckResult(name, False, f"{label} returned empty response")
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
    cfg: dict[str, Any] | None = None,
    verbose: bool = False,
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
    cfg:
        Optional resolved config dict. When provided, budget consistency
        checks are run against the discovered context window.
    verbose:
        Enable litellm debug logging for LLM connectivity checks.
    """
    results: list[CheckResult] = []

    results.append(_check_sources_dir(sources_dir))
    results.append(_check_output_dir(output_dir))
    results.append(_check_parsers())
    results.append(_check_git_repo())
    results.append(_check_disk_space(sources_dir))
    results.append(_check_context_window(default_model, "llm"))
    results.append(_check_context_window(vlm_model, "vlm"))
    results.append(_check_budget_consistency(default_model, cfg))
    llm_timeout = float(cfg.get("compile.llm.timeout", 300)) if cfg else 300.0
    _retries = cfg.get(COMPILE_LLM_OUTPUT_RETRIES) if cfg else None
    results.append(_ping_model(default_model, "llm", verbose=verbose, timeout=llm_timeout, retries=_retries))
    results.append(_ping_model(vlm_model, "vlm", verbose=verbose, timeout=llm_timeout, retries=_retries))
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
