"""``kompend doctor`` — validate configuration and connectivity."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from time import monotonic
from typing import Any

import ollama

from kompend.agents.doctor import DoctorAgent, DoctorDeps
from kompend.cli.cmds.quality._capabilities import (
    measure_compact_formats,
    probe_compact_format,
    probe_native_structured_output,
    probe_prompted_structured_output,
    probe_raw_json,
)
from kompend.core.config import LLMConfig, LlmDataFormat
from kompend.core.config.keys import COMPILE_LLM_CONTEXT_WINDOW, COMPILE_LLM_OUTPUT_RETRIES
from kompend.core.verbosity import is_verbose
from kompend.llm import catalog
from kompend.llm.context_window import discover_context_window
from kompend.llm.models import build_model, is_known_provider, requires_output_ceiling, resolve_model_limits, resolve_vlm_model
from kompend.llm.usage import UsageTracker, tokens_per_second
from kompend.logging import get_console, get_logger
from kompend.preprocessing.parsers._registry import get_registered_parsers
from kompend.rag.zvec_store import build_embedder, to_pydantic_ai_embedding_string
from kompend.types import JSON

#: Smallest input budget a window may leave before the doctor calls it unusable.
_MIN_INPUT_TOKENS: int = 2048

#: Models whose Ollama offload ratio is below this are treated as CPU/partial
#: — typically iGPU-only or VRAM overflow. Below this, inference is slow and
#: the user almost certainly wants to know.
_OLLAMA_GPU_OFFLOAD_MIN = 0.9

logger = get_logger(__name__)


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


def _check_catalog() -> CheckResult:
    """Report the models.dev catalogue and refresh it when it is stale.

    The catalogue answers three questions kompend asks at build time:
    whether a provider is known, which API base URL and env var it uses, and
    how large a model's context and output limits are. ``llmdb`` downloads
    the payload instead of shipping it, so kompend caches it (see
    :mod:`kompend.llm.catalog`) and this check is the only place that
    refreshes an old copy.
    """
    name = "model-catalogue"
    try:
        refreshed = catalog.ensure_fresh()
        current = catalog.snapshot()
    except Exception as exc:
        return CheckResult(name, False, f"Model catalogue unavailable: {exc}")
    if current is None:
        return CheckResult(
            name,
            False,
            "Model catalogue unavailable: download failed and no cached copy exists. Providers are then resolved from [providers.<name>] only.",
            warning=True,
        )
    age = catalog.age()
    age_text = "just now" if age is None else f"{age.total_seconds() / 3600:.1f} h old"
    verb = "refreshed" if refreshed else "cached"
    return CheckResult(
        name,
        True,
        f"{len(current.providers)} providers ({verb}, {age_text}, TTL {catalog.CACHE_TTL_HOURS} h)",
    )


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
    llm_timeout = float(cfg.get("compile.llm.timeout", 300)) if cfg else 300.0
    _retries = cfg.get(COMPILE_LLM_OUTPUT_RETRIES) if cfg else None
    context_window_cap = _cfg_get_int(cfg, COMPILE_LLM_CONTEXT_WINDOW, 0) if cfg else None
    if context_window_cap == 0:
        context_window_cap = None
    llm_cfg = LLMConfig(**(cfg or {}).get("compile", {}).get("llm", {}))

    # Model-touching checks run in per-model blocks (llm → vlm → embed) so a
    # model is never swapped out between its own checks — a reload would
    # otherwise dominate the reported throughput as a bogus cold start.
    # Within a block the connectivity ping comes first: it loads the model
    # into Ollama so /api/ps returns the context the server actually serves
    # (often memory-capped below the GGUF-metadata value).

    results.append(_ping_model(default_model, "llm", verbose=verbose, timeout=llm_timeout, retries=_retries))
    results.extend(
        _check_model_capabilities(
            default_model,
            "llm",
            timeout=llm_timeout,
            retries=_retries,
            data_format=llm_cfg.data_format,
        )
    )
    size_probe = measure_compact_formats()
    results.append(CheckResult(size_probe.name, size_probe.ok, size_probe.detail))
    results.append(_check_context_window(default_model, "llm", context_window_cap, llm_cfg))
    results.append(_check_window_fit(default_model, context_window_cap, llm_cfg))
    results.append(_check_output_ceiling(default_model, "llm"))

    vlm_effective = resolve_vlm_model(default_model, vlm_model)
    if vlm_model:
        results.append(_ping_model(vlm_model, "vlm", verbose=verbose, timeout=llm_timeout, retries=_retries))
    elif vlm_effective:
        # No explicit VLM, but the default model is vision-capable and is used
        # as captioning model. Plain informational note — not a warning: a
        # single configured model is an intentional setup, and the default
        # model's connectivity is already covered by the "llm" ping above.
        results.append(
            CheckResult(
                "connectivity-vlm",
                True,
                f"No vlm model configured — using default ({vlm_effective}) as VLM (vision-capable)",
            )
        )
    else:
        results.append(_ping_model(None, "vlm", verbose=verbose, timeout=llm_timeout, retries=_retries))
    results.extend(_check_model_capabilities(vlm_model, "vlm", timeout=llm_timeout, retries=_retries, full=False))
    results.append(_check_context_window(vlm_model, "vlm", context_window_cap, llm_cfg))

    referenced_providers = {provider_id for model in (default_model, vlm_model, embed_model) if model and (provider_id := _provider_id_of(model)) is not None}
    results.append(_check_custom_providers(cfg, referenced_providers))
    results.append(_check_ollama_device(default_model, vlm_model))
    # Embed last: loading it evicts the llm/vlm from Ollama, so it must not
    # sit between the llm/vlm checks (nor before ollama-device reads /api/ps).
    results.append(_ping_embed_model(embed_model))

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
    """Check parser availability via the shared registry."""
    name = "parsers"
    return CheckResult(name, True, f"{len(get_registered_parsers())} parser(s) registered")


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


def _check_output_ceiling(model: str | None, label: str) -> CheckResult:
    """Report whether the provider protocol demands an explicit output ceiling.

    kompend sends no ``max_tokens`` for OpenAI-compatible endpoints or Ollama.
    The Anthropic protocol rejects an unset value (pydantic-ai then uses 4096),
    so there kompend computes one per request: window minus the prompt, minus a
    margin, clamped by the model's own output limit when the catalog knows it.
    """
    name = f"output-ceiling-{label}"
    if not model:
        return CheckResult(name, True, f"No {label} model configured - skipped")
    try:
        native = build_model(model)
    except Exception as exc:  # noqa: BLE001 - an unbuildable model is a fact to report
        return CheckResult(name, False, f"Cannot build the model: {exc}")

    if not requires_output_ceiling(native):
        return CheckResult(name, True, "No ceiling sent - the provider default applies")

    try:
        _, limit = resolve_model_limits(model)
    except (KeyError, LookupError, ValueError):
        limit = None
    clamp = f"clamped by the model's {limit}-token output limit" if limit else "catalog knows no output limit for this model"
    return CheckResult(name, True, f"Protocol requires max_tokens: computed per request (window - prompt - margin), {clamp}")


def _check_window_fit(
    model: str | None,
    context_window_cap: int | None = None,
    llm_cfg: LLMConfig | None = None,
) -> CheckResult:
    """Check that the discovered window still leaves room for a request.

    kompend sends no ``max_tokens``: the answer is bounded by the window minus
    its prompt, and the planners reserve a share of the window for it
    (``ContextWindow.answer_reserve``). The only thing that can be wrong is a
    window too small to carry both.
    """
    name = "window-fit"
    if not model:
        return CheckResult(name, True, "No model - skipped")

    try:
        cw = discover_context_window(model, context_window_cap=context_window_cap, llm_config=llm_cfg)
    except Exception as exc:
        return CheckResult(name, False, f"Cannot discover context window: {exc}")

    reserve = cw.answer_reserve
    input_budget = cw.input_budget()
    if input_budget < _MIN_INPUT_TOKENS:
        return CheckResult(
            name,
            False,
            f"Window {cw.max_input} tokens leaves {input_budget} for input after the {reserve}-token answer reserve - increase the model's context window",
        )
    return CheckResult(name, True, f"{cw.max_input} ctx: {reserve} answer reserve, {input_budget} input")


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
        tracker = UsageTracker()
        agent = DoctorAgent(
            model_string=model,
            timeout=timeout,
            retries=retries if retries is not None else 2,
            usage_tracker=tracker,
        )
        deps = DoctorDeps(ping="Respond with exactly one word: ok")
        # Warm-up run: the first call pays model load (cold start) and would
        # otherwise dominate the reported throughput. Discard it and measure
        # the second, warm run — tok/s should reflect steady-state, not load.
        agent.run(deps)
        tokens_before = tracker.completion_tokens
        start = monotonic()
        result = agent.run(deps)
        elapsed = monotonic() - start
        rate = tokens_per_second(tracker.completion_tokens - tokens_before, elapsed)
        rate_s = f" ({rate:.1f} tok/s over {elapsed:.1f}s)" if rate else ""
        if result and result.strip():
            return CheckResult(name, True, f"{label} OK ({model}){rate_s}")
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


def _check_model_capabilities(
    model: str | None,
    label: str,
    *,
    timeout: float = 120.0,
    retries: int | None = None,
    data_format: LlmDataFormat = LlmDataFormat.JSON,
    full: bool = True,
) -> list[CheckResult]:
    """Probe how the model produces structured output.

    The first probe — native tool output — is the capability every typed
    kompend agent depends on. The extra probes tell the user what else the
    model manages: a model that fails natively but answers a schema in the
    prompt is still usable, and a model that can emit the configured
    compact format saves context on every structured payload.
    """
    if not model:
        return [CheckResult(f"structured-output-{label}", True, f"No {label} model configured — skipped")]
    budget = retries if retries is not None else 1
    probes = [probe_native_structured_output(model, timeout=timeout, retries=budget)]
    if full:
        probes.append(probe_prompted_structured_output(model, timeout=timeout, retries=budget))
        probes.append(probe_raw_json(model, timeout=timeout, retries=budget))
        if data_format in (LlmDataFormat.TOON, LlmDataFormat.ISON):
            probes.append(probe_compact_format(model, data_format, timeout=timeout, retries=budget))
    results: list[CheckResult] = []
    for probe in probes:
        check_name = f"structured-output-{label}" if probe.name == "structured-output" else f"{probe.name}-{label}"
        results.append(CheckResult(check_name, probe.ok, f"{probe.detail} (model={model})", warning=probe.skipped))
    return results


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
            # Override-only block for a known provider (e.g. [providers.ollama]
            # with only context_window / max_connections) — settings apply,
            # model building still routes through the native/catalog path.
            # Standalone providers (typo, never-known id) without base_url
            # remain a hard error.
            if is_known_provider(prov_name):
                continue
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


def _check_ollama_device(default_model: str | None, vlm_model: str | None) -> CheckResult:
    """Report where Ollama loaded each referenced model (GPU / partial / CPU).

    Ollama exposes only ``size_vram`` vs ``size`` — there's no per-engine field.
    The ratio is a reliable proxy: full offload means a real GPU is in use;
    partial typically means iGPU/overflow (slow).

    Roles that resolve to the same loaded instance (same digest — e.g. one
    vision-capable model acting as both llm and vlm) share one merged
    ``llm+vlm:`` line, because Ollama loads and bills that VRAM once.
    """
    name = "ollama-device"
    targets = [("llm", default_model), ("vlm", vlm_model)]
    targets = [(label, m) for label, m in targets if m and _provider_id_of(m) == "ollama"]  # nocheck: enum-literal — provider id, not the judge vocabulary
    if not targets:
        return CheckResult(name, True, "No ollama/ models configured - skipped")

    # Group by loaded identity: two roles pointing at the same model (same
    # digest, e.g. a vision-capable default serving as VLM) resolve to one
    # loaded Ollama instance, so they share a merged ``llm+vlm`` line instead
    # of two identical VRAM reports that read as double the memory. Unloaded
    # models group by their own name so distinct not-loaded variants stay
    # separate.
    labels_by_key: dict[str, list[str]] = {}
    entry_by_key: dict[str, dict[str, Any] | None] = {}
    for label, model in targets:
        bare = model.split("/", 1)[1] if "/" in model else model
        entry = _ollama_ps_lookup(bare)
        key = entry["digest"] if entry is not None and entry.get("digest") else f"unloaded:{bare}"
        labels_by_key.setdefault(key, []).append(label)
        entry_by_key[key] = entry

    parts: list[str] = []
    any_warn = False
    for key, entry in entry_by_key.items():
        merged = "+".join(labels_by_key[key])
        if entry is None:
            parts.append(f"{merged}: not loaded")
            continue
        ratio = entry["size_vram"] / entry["size"] if entry["size"] else 0.0
        kind = _classify_ollama_offload(entry["size"], entry["size_vram"])
        vram_mib = entry["size_vram"] / (1024 * 1024)
        size_mib = entry["size"] / (1024 * 1024)
        parts.append(f"{merged}: {kind} ({vram_mib:.0f}/{size_mib:.0f} MiB VRAM, {ratio:.0%})")
        if kind != "GPU":
            any_warn = True

    nvidia = _nvidia_smi_probe()
    if nvidia and any_warn:
        used = nvidia["memory_used_mib"]
        total = nvidia["memory_total_mib"]
        free = total - used
        parts.append(
            f"NVIDIA {nvidia['name']} detected ({free}/{total} MiB free) but Ollama is not using it — check OLLAMA_NUM_GPU / drivers / model fits in VRAM"
        )
    elif nvidia and not any_warn:
        parts.append(f"NVIDIA {nvidia['name']} ({nvidia['memory_total_mib']} MiB) — Ollama is using it")

    msg = "; ".join(parts)
    if any_warn:
        return CheckResult(name, True, msg, warning=True)
    return CheckResult(name, True, msg)


def _provider_id_of(model_string: str) -> str | None:
    """Extract the provider prefix from a ``provider/model`` string."""
    if "/" in model_string:
        return model_string.split("/", 1)[0]
    if ":" in model_string:
        return model_string.split(":", 1)[0]
    return None


def _ollama_ps_lookup(model_name: str) -> dict[str, Any] | None:
    """Find ``model_name`` in ``ollama.ps()`` output.

    An exact match on the full name (including ``:tag``) wins. Only a name
    without a tag (registry default implied) may fall back to matching any
    loaded model of the same base name. An explicitly tagged name that is
    not loaded returns ``None`` — it must not absorb the stats of a
    different tag of the same base model (e.g. ``qwen3:4b`` configured
    while ``qwen3:8b`` is loaded would falsely report 8b's VRAM as 4b's).

    Returns a plain dict with ``size``, ``size_vram``, ``context_length``,
    ``digest`` so the result is JSON-serializable.
    """
    try:
        ps_models = ollama.ps().models
    except Exception as exc:
        logger.debug("ollama.ps() failed for device check: %s", exc, exc_info=is_verbose())
        return None

    def _entry(m: Any) -> dict[str, Any]:
        return {
            "size": int(m.size or 0),
            "size_vram": int(m.size_vram or 0),
            "context_length": int(m.context_length or 0),
            "digest": m.digest,
        }

    tagged = ":" in model_name
    for m in ps_models:
        for ps_id in (m.model, m.name):
            if not ps_id:
                continue
            if ps_id == model_name:
                return _entry(m)
            if not tagged and ps_id.split(":", 1)[0] == model_name:
                return _entry(m)
    return None


def _classify_ollama_offload(size: int, size_vram: int) -> str:
    """Return a one-word label for the offload ratio."""
    if size <= 0:
        return "unknown"
    ratio = size_vram / size
    if ratio >= 0.99:
        return "GPU"
    if ratio >= _OLLAMA_GPU_OFFLOAD_MIN:
        return "GPU"  # marginal — still mostly on VRAM
    if ratio > 0.0:
        return "partial"  # split across VRAM and system RAM (iGPU / overflow)
    return "CPU"  # 0% offload — CPU only


def _nvidia_smi_probe() -> dict[str, Any] | None:
    """Return a short NVIDIA GPU report from ``nvidia-smi``, or None.

    Used to flag the case where Ollama didn't offload to VRAM despite an
    NVIDIA GPU being present (driver mismatch, OLLAMA_NUM_GPU unset, etc.).
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(  # noqa: S603 — args are hardcoded, smi path from shutil.which
            [smi, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    line = out.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        return None
    try:
        return {"name": parts[0], "memory_total_mib": int(parts[1]), "memory_used_mib": int(parts[2])}
    except ValueError:
        return None


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
