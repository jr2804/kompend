"""Provider checks: the model catalogue, custom endpoints, Ollama's offload.

Model-string parsing (``provider_id_of``/``model_slug``) lives in
:mod:`kompend.llm.model_id` and is imported here, so the doctor's provider
lookups and its Ollama calls cannot disagree about which provider a model
belongs to.
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Final, Protocol, cast

from kompend.core.config import ProviderEndpoint
from kompend.diagnostics.base import CheckResult
from kompend.llm import catalog, model_slug, provider_id_of
from kompend.llm.models import is_known_provider
from kompend.logging import get_logger

#: The provider id Ollama models carry in ``provider/model`` strings.
OLLAMA_PROVIDER_ID: Final[str] = "ollama"

#: Below this VRAM share the model is treated as CPU/partial — typically an
#: iGPU or a VRAM overflow. Inference is slow there and the user almost
#: certainly wants to know.
OLLAMA_GPU_OFFLOAD_MIN: Final[float] = 0.9

#: Seconds a ``[providers.<name>]`` endpoint gets to answer ``/models``. Short
#: on purpose: the doctor must not hang on an unreachable host.
PROVIDER_PROBE_TIMEOUT_S: Final[float] = 5.0

#: Seconds ``nvidia-smi`` gets. It answers from a local driver, so anything
#: slower means it is stuck.
NVIDIA_SMI_TIMEOUT_S: Final[int] = 10

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class OllamaLoadedModel:
    """One model Ollama has in memory, as the doctor reports it."""

    size: int
    size_vram: int
    context_length: int
    digest: str


@dataclass(frozen=True, slots=True)
class NvidiaReport:
    """One GPU as ``nvidia-smi`` reports it."""

    name: str
    memory_total_mib: int
    memory_used_mib: int


@dataclass(frozen=True, slots=True)
class OllamaModelIdentity:
    """The weights behind one Ollama tag, as ``ollama show`` reports them.

    ``digest`` names what the tag actually serves: an alias tag (a second
    layer built ``FROM`` an existing model) reports its parent's digest, so
    two tags that share a digest produce byte-identical outputs even though
    their names — and kompend's index identity — differ. A tag with no parent
    falls back to its own digest from ``ollama list``.
    """

    digest: str | None
    parameter_size: str | None
    native_dim: int | None


class _LoadedModel(Protocol):
    """The fields the doctor reads off an ``ollama.ps()`` entry."""

    size: int | None
    size_vram: int | None
    context_length: int | None
    digest: str | None


def check_catalogue() -> CheckResult:
    """Report the models.dev catalogue and refresh it when it is stale.

    The catalogue answers three questions kompend asks at build time: whether a
    provider is known, which API base URL and env var it uses, and how large a
    model's context and output limits are. ``llmdb`` downloads the payload
    instead of shipping it, so kompend caches it (see
    :mod:`kompend.llm.catalog`) and this check is the only place that refreshes
    an old copy.
    """
    name = "model-catalogue"
    try:
        refreshed = catalog.ensure_fresh()
        current = catalog.snapshot()
    except Exception as exc:  # noqa: BLE001 — an unusable catalogue is the fact being reported
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


def check_custom_providers(
    providers: Mapping[str, ProviderEndpoint],
    referenced_providers: set[str] | None = None,
) -> CheckResult:
    """Validate the ``[providers.<name>]`` endpoints.

    Only providers actually referenced by a configured model string are
    checked — a configured-but-unused provider is skipped (its host may
    legitimately be down). A block without ``base_url`` is legal for an
    integrated provider (an override block, e.g. ``[providers.ollama]`` with
    only ``context_window``); for an unknown name it is a typo and fails.
    """
    name = "custom-providers"
    if not providers:
        return CheckResult(name, True, "No custom providers configured")
    checked: list[str] = []
    failures: list[str] = []
    for provider_name, endpoint in providers.items():
        if referenced_providers is not None and provider_name not in referenced_providers:
            continue
        checked.append(provider_name)
        if not endpoint.base_url:
            if is_known_provider(provider_name):
                continue
            failures.append(f"{provider_name}: missing base_url")
            continue
        request = urllib.request.Request(endpoint.base_url.rstrip("/") + "/models")  # noqa: S310 — the URL is the operator's own [providers.<name>] entry
        if endpoint.api_key:
            request.add_header("Authorization", f"Bearer {endpoint.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=PROVIDER_PROBE_TIMEOUT_S) as response:  # noqa: S310 — see above
                response.read(1)
        except urllib.error.HTTPError as exc:
            if exc.code not in (401, 404):
                # 401/404 mean reachable: it needs auth, or has no /models route.
                failures.append(f"{provider_name}: HTTP {exc.code} at {endpoint.base_url}")
        except OSError as exc:
            failures.append(f"{provider_name}: unreachable ({exc})")
    if failures:
        return CheckResult(name, False, "; ".join(failures))
    if not checked:
        return CheckResult(name, True, "No custom providers in use — skipping")
    return CheckResult(name, True, f"{len(checked)} provider(s) in use OK")


def check_ollama_device(default_model: str | None, vlm_model: str | None) -> CheckResult:
    """Report where Ollama loaded each referenced model (GPU / partial / CPU).

    Ollama exposes only ``size_vram`` vs ``size`` — there is no per-engine
    field. The ratio is a reliable proxy: full offload means a real GPU is in
    use; partial typically means iGPU/overflow (slow).

    Roles that resolve to the same loaded instance (same digest — e.g. one
    vision-capable model acting as both llm and vlm) share one merged
    ``llm+vlm:`` line, because Ollama loads and bills that VRAM once.
    """
    name = "ollama-device"
    targets = [("llm", default_model), ("vlm", vlm_model)]
    targets = [(label, model) for label, model in targets if model and provider_id_of(model) == OLLAMA_PROVIDER_ID]
    if not targets:
        return CheckResult(name, True, "No ollama/ models configured - skipped")

    # Group by loaded identity: two roles pointing at the same model (same
    # digest, e.g. a vision-capable default serving as VLM) resolve to one
    # loaded Ollama instance, so they share a merged ``llm+vlm`` line instead
    # of two identical VRAM reports that read as double the memory. Unloaded
    # models group by their own name so distinct not-loaded variants stay
    # separate.
    labels_by_key: dict[str, list[str]] = {}
    entry_by_key: dict[str, OllamaLoadedModel | None] = {}
    for label, model in targets:
        bare = model_slug(model)
        entry = ollama_ps_lookup(bare)
        key = entry.digest if entry is not None and entry.digest else f"unloaded:{bare}"
        labels_by_key.setdefault(key, []).append(label)
        entry_by_key[key] = entry

    parts: list[str] = []
    any_warn = False
    for key, entry in entry_by_key.items():
        merged = "+".join(labels_by_key[key])
        if entry is None:
            parts.append(f"{merged}: not loaded")
            continue
        kind = classify_ollama_offload(entry.size, entry.size_vram)
        vram_mib = entry.size_vram / (1024 * 1024)
        size_mib = entry.size / (1024 * 1024)
        ratio = entry.size_vram / entry.size if entry.size else 0.0
        parts.append(f"{merged}: {kind} ({vram_mib:.0f}/{size_mib:.0f} MiB VRAM, {ratio:.0%})")
        if kind != "GPU":
            any_warn = True

    nvidia = nvidia_smi_probe()
    if nvidia and any_warn:
        free = nvidia.memory_total_mib - nvidia.memory_used_mib
        parts.append(
            f"NVIDIA {nvidia.name} detected ({free}/{nvidia.memory_total_mib} MiB free) but Ollama is not using it — "
            f"check OLLAMA_NUM_GPU / drivers / model fits in VRAM"
        )
    elif nvidia:
        parts.append(f"NVIDIA {nvidia.name} ({nvidia.memory_total_mib} MiB) — Ollama is using it")

    message = "; ".join(parts)
    if any_warn:
        return CheckResult(name, True, message, warning=True)
    return CheckResult(name, True, message)


def ollama_ps_lookup(model_name: str) -> OllamaLoadedModel | None:
    """Find ``model_name`` in ``ollama.ps()`` output.

    An exact match on the full name (including ``:tag``) wins. Only a name
    without a tag (registry default implied) may fall back to matching any
    loaded model of the same base name. An explicitly tagged name that is not
    loaded returns ``None`` — it must not absorb the stats of a different tag
    of the same base model (e.g. ``qwen3:4b`` configured while ``qwen3:8b`` is
    loaded would falsely report 8b's VRAM as 4b's).
    """
    ollama = import_module("ollama")

    try:
        ps_models = ollama.ps().models
    except Exception as exc:  # noqa: BLE001 — an unreachable daemon means "not loaded"
        logger.debug("ollama.ps() failed for device check: %s", exc)
        return None

    tagged = ":" in model_name
    for loaded in ps_models:
        for loaded_id in (loaded.model, loaded.name):
            if not loaded_id:
                continue
            if loaded_id == model_name:
                return _as_loaded_model(loaded)
            if not tagged and loaded_id.split(":", 1)[0] == model_name:
                return _as_loaded_model(loaded)
    return None


def _as_loaded_model(loaded: object) -> OllamaLoadedModel:
    """Read one ``ollama.ps()`` entry into the doctor's own record."""
    entry = cast("_LoadedModel", loaded)
    return OllamaLoadedModel(
        size=int(entry.size or 0),
        size_vram=int(entry.size_vram or 0),
        context_length=int(entry.context_length or 0),
        digest=entry.digest or "",
    )


def ollama_show_lookup(model_name: str) -> OllamaModelIdentity | None:
    """Resolve one Ollama tag to the weights it serves, or ``None``.

    The digest prefers ``details.parent_model``: that is what makes alias tags
    visible — ``embeddinggemma-2:440m`` built ``FROM`` ``embeddinggemma-2:270m``
    reports the base model's digest, so the doctor can tell "different tags,
    same weights" from a genuinely different model. Without a reachable daemon
    there is no identity; the caller degrades to the plain tag name.
    """
    ollama = import_module("ollama")

    try:
        shown = ollama.show(model_name)
    except Exception as exc:  # noqa: BLE001 — an unreachable daemon means no identity
        logger.debug("ollama.show(%s) failed: %s", model_name, exc)
        return None
    digest = None
    if shown.details is not None and shown.details.parent_model:
        digest = shown.details.parent_model
    if digest is None:
        digest = _ollama_list_digest(model_name)
    native_dim = None
    for key, value in sorted((shown.modelinfo or {}).items()):
        if key.endswith(".embedding_length") and isinstance(value, int):
            native_dim = value
            break
    parameter_size = shown.details.parameter_size if shown.details is not None else None
    return OllamaModelIdentity(digest=digest, parameter_size=parameter_size, native_dim=native_dim)


def _ollama_list_digest(model_name: str) -> str | None:
    """The manifest digest of ``model_name`` from ``ollama list``, or ``None``.

    ``ollama show`` reports a parent digest only for alias tags; a tag with no
    parent has no ``details.parent_model``, so its own digest comes from the
    local manifest instead.
    """
    ollama = import_module("ollama")

    try:
        listing = ollama.list()
    except Exception as exc:  # noqa: BLE001 — an unreachable daemon means no digest
        logger.debug("ollama.list() failed: %s", exc)
        return None
    for entry in getattr(listing, "models", ()) or ():
        if getattr(entry, "model", None) == model_name:
            return getattr(entry, "digest", None) or None
    return None


def classify_ollama_offload(size: int, size_vram: int) -> str:
    """Return a one-word label for the offload ratio."""
    if size <= 0:
        return "unknown"
    if size_vram / size >= OLLAMA_GPU_OFFLOAD_MIN:
        return "GPU"
    if size_vram > 0:
        return "partial"  # split across VRAM and system RAM (iGPU / overflow)
    return "CPU"  # 0% offload — CPU only


def nvidia_smi_probe() -> NvidiaReport | None:
    """Return a short NVIDIA GPU report from ``nvidia-smi``, or ``None``.

    Used to flag the case where Ollama did not offload to VRAM despite an
    NVIDIA GPU being present (driver mismatch, ``OLLAMA_NUM_GPU`` unset, …).
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 — the arguments are hardcoded, the path comes from shutil.which
            [smi, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    parts = [part.strip() for part in completed.stdout.strip().splitlines()[0].split(",")]
    if len(parts) != 3:
        return None
    try:
        return NvidiaReport(name=parts[0], memory_total_mib=int(parts[1]), memory_used_mib=int(parts[2]))
    except ValueError:
        return None
