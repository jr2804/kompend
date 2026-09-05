"""``knox doctor`` — validate configuration and connectivity."""

from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from time import monotonic
from typing import Any

from pydantic import BaseModel as _BM
from pydantic_ai import Agent as _Agent
from pydantic_ai.settings import ModelSettings

from knox.agents.doctor import DoctorAgent, DoctorDeps
from knox.core.config import LLMConfig
from knox.core.config.keys import COMPILE_LLM_CONTEXT_WINDOW, COMPILE_LLM_CONTEXT_WINDOW_CAP, COMPILE_LLM_OUTPUT_RETRIES
from knox.llm.context_window import discover_context_window
from knox.llm.models import build_model
from knox.logging import get_console, get_logger
from knox.preprocessing.parsers import build_default_router
from knox.rag.zvec_store import build_embedder, to_pydantic_ai_embedding_string
from knox.types import JSON

logger = get_logger(__name__)


class _CapabilityCheck(_BM):
    """Structured output schema for the structured-output doctor check."""

    ok: bool
    message: str


# ── Check results ───────────────────────────────────────────────────────


class CheckResult:
    """Result of a single doctor check."""

    def __init__(self, name: str, passed: bool, message: str = "", warning: bool = False) -> None:
        self.name = name
        self.passed = passed
        self.message = message
        self.warning = warning

    def to_dict(self) -> dict[str, str | bool]:
        return {"name": self.name, "passed": self.passed, "message": self.message, "warning": self.warning}


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
        LLM model name (``provider/model`` format) for the connectivity ping.
    vlm_model:
        VLM model name for the connectivity ping.
    embed_model:
        Embedding model name for the connectivity ping.
    cfg:
        Optional resolved config dict. When provided, budget consistency
        checks are run against the discovered context window.
    verbose:
        Enable verbose debug logging for LLM connectivity checks.
    """
    results: list[CheckResult] = []

    results.append(_check_sources_dir(sources_dir))
    results.append(_check_output_dir(output_dir))
    results.append(_check_parsers())
    results.append(_check_git_repo())
    results.append(_check_disk_space(sources_dir))
    context_window_cap = _cfg_get_int(cfg, COMPILE_LLM_CONTEXT_WINDOW, 0) if cfg else None
    if context_window_cap is None:
        context_window_cap = _cfg_get_int(cfg, COMPILE_LLM_CONTEXT_WINDOW_CAP, 0) if cfg else None
    if context_window_cap == 0:
        context_window_cap = None
    llm_cfg = LLMConfig(**(cfg or {}).get("compile", {}).get("llm", {}))
    results.append(_check_context_window(default_model, "llm", context_window_cap, llm_cfg))
    results.append(_check_context_window(vlm_model, "vlm", context_window_cap, llm_cfg))
    results.append(_check_budget_consistency(default_model, cfg, context_window_cap))
    llm_timeout = float(cfg.get("compile.llm.timeout", 300)) if cfg else 300.0
    _retries = cfg.get(COMPILE_LLM_OUTPUT_RETRIES) if cfg else None
    results.append(_ping_model(default_model, "llm", verbose=verbose, timeout=llm_timeout, retries=_retries))
    results.append(_ping_model(vlm_model, "vlm", verbose=verbose, timeout=llm_timeout, retries=_retries))
    results.append(_ping_embed_model(embed_model))
    results.append(_check_structured_output(default_model, "llm", timeout=llm_timeout, retries=_retries))
    results.append(_check_structured_output(vlm_model, "vlm", timeout=llm_timeout, retries=_retries))
    referenced_providers = {provider_id for model in (default_model, vlm_model, embed_model) if model and (provider_id := _provider_id_of(model)) is not None}
    results.append(_check_custom_providers(cfg, referenced_providers))

    return results


# ── Checks ──────────────────────────────────────────────────────────────


def _check_sources_dir(sources_dir: Path | None) -> CheckResult:
    """Validate that the sources directory exists and is a directory."""
    name = "sources-directory"
    if sources_dir is None:
        return CheckResult(name, True, "No sources dir configured — skipped")
    if not sources_dir.exists():
        return CheckResult(name, False, f"Sources dir does not exist: {sources_dir}")
    if not sources_dir.is_dir():
        return CheckResult(name, False, f"Sources path is not a directory: {sources_dir}")
    return CheckResult(name, True, f"Sources dir OK: {sources_dir}")


def _check_output_dir(output_dir: Path | None) -> CheckResult:
    """Validate that the output directory exists and is writable."""
    name = "output-directory"
    if output_dir is None:
        return CheckResult(name, True, "No output dir configured — skipped")
    if not output_dir.exists():
        return CheckResult(name, False, f"Output dir does not exist: {output_dir}")
    if not os.access(output_dir, os.W_OK):
        return CheckResult(name, False, f"Output dir not writable: {output_dir}")
    return CheckResult(name, True, f"Output dir OK: {output_dir}")


def _check_parsers() -> CheckResult:
    """Check parser availability via the shared registry factory."""
    name = "parsers"

    router = build_default_router()
    return CheckResult(name, True, f"{len(router)} parser(s) registered")


def _check_context_window(
    model: str | None,
    label: str,
    context_window_cap: int | None = None,
    llm_config: LLMConfig | None = None,
) -> CheckResult:
    """Discover and report the context window for a model."""
    name = f"context-window-{label}"
    if not model:
        return CheckResult(name, True, f"No {label} model configured — skipped")

    try:
        cw = discover_context_window(model, llm_config=llm_config, context_window_cap=context_window_cap)
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


def _check_budget_consistency(model: str | None, cfg: dict[str, Any] | None, context_window_cap: int | None = None) -> CheckResult:
    """Check that budget config values don't exceed the model's context window."""
    name = "budget-consistency"
    if not model or cfg is None:
        return CheckResult(name, True, "No model or config — skipped")

    try:
        llm_cfg = LLMConfig(**(cfg or {}).get("compile", {}).get("llm", {}))
        cw = discover_context_window(model, llm_config=llm_cfg, context_window_cap=context_window_cap)
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
            "Reduce gen_output_max or increase context_window."
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
    val: JSON = cfg
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


def _ping_model(model: str | None, label: str, verbose: bool = False, timeout: float = 60.0, retries: int | None = None) -> CheckResult:
    """Ping a single completion-type model (LLM or VLM) via PydanticAI DoctorAgent."""
    name = f"connectivity-{label}"
    if not model:
        if label == "llm":
            # Missing LLM is a hard failure — a compile cannot run without it.
            return CheckResult(name, False, "No llm model configured — skipped")
        return CheckResult(name, True, f"No {label} model configured — skipped", warning=True)

    try:
        agent = DoctorAgent(
            model_string=model,
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
    """Ping an embedding model via pydantic-ai's Embedder."""
    name = "connectivity-embed"
    if not model:
        return CheckResult(name, True, "No embed model configured — skipped", warning=True)

    try:
        pydantic_ai_string = to_pydantic_ai_embedding_string(model)
        embedder = build_embedder(pydantic_ai_string)
        result = embedder.embed_documents_sync("ping")
        dims = len(result.embeddings[0]) if result.embeddings else 0
        return CheckResult(name, True, f"embed OK ({pydantic_ai_string}, {dims} dims)")
    except Exception as exc:
        return CheckResult(name, False, f"embed failed: {exc}")


def _check_structured_output(
    model: str | None,
    label: str,
    timeout: float = 120.0,
    retries: int | None = None,
) -> CheckResult:
    """Probe a model's ability to produce valid structured JSON output.

    Creates a trivial PydanticAI agent with a simple schema and checks
    whether the model can return valid structured data. This catches the
    "model returns empty string" failure mode that small or unsuitable
    models exhibit.
    """
    name = f"structured-output-{label}"
    if not model:
        return CheckResult(name, True, f"No {label} model configured — skipped")

    try:
        agent = _Agent(
            build_model(model),
            output_type=_CapabilityCheck,
            retries=retries or 1,
            model_settings=ModelSettings(timeout=timeout, max_tokens=128),
            name=f"doctor-structured-output-{label}",
        )
        start = monotonic()
        result = agent.run_sync("Respond with a JSON object where ok=true and message='hello'")
        elapsed = monotonic() - start
        validated = result.output
        if isinstance(validated, _CapabilityCheck) and validated.ok:
            return CheckResult(
                name,
                True,
                f"{label} OK ({elapsed:.1f}s, model={model})",
            )
        return CheckResult(
            name,
            False,
            f"{label} returned invalid structured output after {elapsed:.1f}s",
        )
    except Exception as exc:
        return CheckResult(name, False, f"{label} structured output test failed: {exc}")


def _check_git_repo() -> CheckResult:
    """Check if we are inside a git repository."""
    name = "git-repo"
    # Traverse up from cwd looking for a .git directory.
    # Pure-Path check avoids subprocess entirely (no S603 false positive).
    try:
        current_dir = Path.cwd().resolve()
        for parent in [current_dir, *current_dir.parents]:
            if (parent / ".git").is_dir():
                return CheckResult(name, True, "Inside a git repository")
        return CheckResult(name, True, "Not a git repository — OK")
    except Exception as exc:
        return CheckResult(name, True, f"Git check failed ({exc}) — OK")


def _provider_id_of(model_string: str) -> str | None:
    """Extract the provider prefix from a ``provider/model`` string."""
    if "/" in model_string:
        return model_string.split("/", 1)[0]
    if ":" in model_string:
        return model_string.split(":", 1)[0]
    return None


def _check_custom_providers(cfg: dict[str, Any] | None, referenced_providers: set[str] | None = None) -> CheckResult:
    """Validate custom providers from [providers.*] sections.

    Only providers actually referenced by a configured model string are
    checked — configured-but-unused providers are skipped (their host may
    legitimately be down).
    """
    name = "Custom providers"
    if not cfg:
        return CheckResult(name, True, "No config — skipping")
    providers = cfg.get("providers", {})
    if not providers:
        return CheckResult(name, True, "No custom providers configured")
    checked: list[str] = []
    failures: list[str] = []
    for prov_name, prov_cfg in providers.items():
        if referenced_providers is not None and prov_name not in referenced_providers:
            continue
        checked.append(prov_name)
        base_url = prov_cfg.get("base_url", "") if isinstance(prov_cfg, dict) else ""
        if not base_url:
            failures.append(f"{prov_name}: missing base_url")
            continue
        # Quick connectivity check — try to reach the base URL
        try:
            req = urllib.request.Request(base_url.rstrip("/") + "/models")  # noqa: S310 — doctor check, user-configured URL
            api_key = prov_cfg.get("api_key", "") if isinstance(prov_cfg, dict) else ""
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            conn = urllib.request.urlopen(req, timeout=5)  # noqa: S310 — doctor check, user-configured URL
            conn.close()
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 404):
                pass  # reachable but needs auth or no /models endpoint
            else:
                failures.append(f"{prov_name}: HTTP {exc.code} at {base_url}")
        except Exception as exc:
            failures.append(f"{prov_name}: unreachable ({exc})")
    if failures:
        return CheckResult(name, False, "; ".join(failures))
    if not checked:
        return CheckResult(name, True, "No custom providers in use — skipping")
    return CheckResult(name, True, f"{len(checked)} provider(s) in use OK")


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
    except Exception as exc:
        return CheckResult(name, True, f"Could not check disk space ({exc}) — skipped")


def print_doctor_results(results: list[CheckResult], json_output: bool = False) -> int:
    """Print doctor results. Returns exit code (0 = no failures)."""
    failed_count = sum(1 for r in results if not r.passed)
    warn_count = sum(1 for r in results if r.passed and r.warning)
    ok_count = sum(1 for r in results if r.passed and not r.warning)

    if json_output:
        data = {
            "passed": ok_count,
            "warning": warn_count,
            "failed": failed_count,
            "total": len(results),
            "checks": [r.to_dict() for r in results],
        }
        json.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for r in results:
            if not r.passed:
                icon = "[!]"
            elif r.warning:
                icon = "[WARN]"
            else:
                icon = "[OK]"
            get_console().print(f"  {icon} {r.name}: {r.message}", markup=False)

        total = len(results)
        get_console().print(f"\n  {ok_count}/{total} checks passed", end="")
        if warn_count:
            get_console().print(f", {warn_count} warning(s)", end="")
        if failed_count:
            get_console().print(f", {failed_count} failed", end="")
        get_console().print("")

    return 1 if failed_count else 0
