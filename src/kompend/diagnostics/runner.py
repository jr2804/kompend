"""The doctor run: which checks run, and in which order.

Order is part of the contract, not an accident of the file layout:

* The catalogue comes first — provider and model resolution read it.
* Workspace checks (directories, git, disk, the parser) cost nothing and fail
  fast, so a misconfigured vault never waits for a model download.
* The model checks run in per-model blocks (llm → vlm → embed) so a model is
  never swapped out between two of its own checks: a reload would otherwise
  show up as a bogus cold start in the reported throughput. The connectivity
  ping opens each block because it loads the model, and Ollama only reports the
  context it actually serves once the model is in memory.
* The embedder runs last: loading it evicts the completion models from VRAM.
"""

from __future__ import annotations

from pathlib import Path

from kompend.core.config import LLMConfig, ResolvedConfig
from kompend.diagnostics.base import CheckResult
from kompend.diagnostics.capabilities import measure_compact_formats
from kompend.diagnostics.environment import (
    check_disk_space,
    check_git_repo,
    check_output_dir,
    check_parsers,
    check_sources_dir,
)
from kompend.diagnostics.models import (
    check_context_window,
    check_model_capabilities,
    check_output_ceiling,
    check_server_window,
    check_window_fit,
    ping_embed_model,
    ping_model,
)
from kompend.diagnostics.providers import (
    check_catalogue,
    check_custom_providers,
    check_ollama_device,
    provider_id_of,
)
from kompend.llm.models import resolve_vlm_model


def run_doctor(
    *,
    sources_dir: Path | None = None,
    output_dir: Path | None = None,
    default_model: str | None = None,
    vlm_model: str | None = None,
    embed_model: str | None = None,
    config: ResolvedConfig | None = None,
    verbose: bool = False,
) -> list[CheckResult]:
    """Run every doctor check and return the results in report order.

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
    config:
        The resolved configuration. When omitted, the LLM defaults apply and no
        ``[providers.<name>]`` block is checked.
    verbose:
        Enable verbose debug logging for the connectivity checks.
    """
    llm = config.kompend.compile.llm if config is not None else LLMConfig()
    providers = config.kompend.providers if config is not None else {}
    results: list[CheckResult] = []

    results.append(check_catalogue())
    results.append(check_sources_dir(sources_dir))
    results.append(check_output_dir(output_dir))
    results.append(check_parsers())
    results.append(check_git_repo())
    results.append(check_disk_space(sources_dir))

    results.append(ping_model(default_model, "llm", verbose=verbose, timeout=llm.timeout, retries=llm.output_retries))
    results.extend(
        check_model_capabilities(
            default_model,
            "llm",
            timeout=llm.timeout,
            retries=llm.output_retries,
            data_format=llm.data_format,
        )
    )
    size_probe = measure_compact_formats()
    results.append(CheckResult(size_probe.name, size_probe.ok, size_probe.detail))
    results.append(check_context_window(default_model, "llm", llm.context_window, llm))
    results.append(check_window_fit(default_model, llm.context_window, llm))
    results.append(check_server_window(default_model, llm.context_window))
    results.append(check_output_ceiling(default_model, "llm"))

    vlm_effective = resolve_vlm_model(default_model, vlm_model)
    if vlm_model:
        results.append(ping_model(vlm_model, "vlm", verbose=verbose, timeout=llm.timeout, retries=llm.output_retries))
    elif vlm_effective:
        # No explicit VLM, but the default model is vision-capable and is used
        # as the captioning model. An informational note, not a warning: a
        # single configured model is an intentional setup, and its connectivity
        # is already covered by the "llm" ping above.
        results.append(
            CheckResult(
                "connectivity-vlm",
                True,
                f"No vlm model configured — using default ({vlm_effective}) as VLM (vision-capable)",
            )
        )
    else:
        results.append(ping_model(None, "vlm", verbose=verbose, timeout=llm.timeout, retries=llm.output_retries))
    results.extend(check_model_capabilities(vlm_model, "vlm", timeout=llm.timeout, retries=llm.output_retries, full=False))
    results.append(check_context_window(vlm_model, "vlm", llm.context_window, llm))

    referenced_providers = {provider for model in (default_model, vlm_model, embed_model) if model and (provider := provider_id_of(model)) is not None}
    results.append(check_custom_providers(providers, referenced_providers))
    results.append(check_ollama_device(default_model, vlm_model))
    results.append(ping_embed_model(embed_model))

    return results


__all__ = ["run_doctor"]
