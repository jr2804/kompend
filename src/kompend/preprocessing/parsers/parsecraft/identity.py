"""The two identity halves of the parsecraft backend.

``backend_identity`` names what can parse here (registered backends, installed
extras, the resolved ``[routing]`` pins); ``cache_identity`` names the routing
inputs that change what the document is parsed into (judge spec, preference,
OCR permission, engines, backend options, image mode). Together they are the
raw material of the parse-cache key — see ``DocumentParser.backend_identity``
(``base.py``), which composes both.

Both functions are pure: they take the probed environment, the loaded
settings and the image store as values, so a refactor of the adapter cannot
silently re-derive them from different inputs. The byte-exact contract for
both strings is pinned in ``tests/unit/test_parsecraft_parser.py``
(``TestIdentityGoldenStrings``).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from parsecraft.environment import EnvironmentInfo
from parsecraft.routing.models import RoutingPreference

from kompend.preprocessing.images import ImageStore
from kompend.preprocessing.parsers.parsecraft_routing import ParsecraftRoutingConfig


def backend_identity(environment: EnvironmentInfo, settings: ParsecraftRoutingConfig) -> str:
    """Which backends can parse here, and which ones the config allows.

    The identity names backends, not machinery: the backends registered on
    this host, the extras that make their capabilities importable, and the
    restrictions the configuration puts on them (``only``/``preferred``
    backend, engine list, OCR permission). The parsecraft version, the
    judge, the preference and the effort are deliberately absent — a
    version bump or an unrelated bug fix must not re-parse a vault, while a
    changed backend set must.

    The restrictions are read off the *resolved* routing config, not off
    kompend's own attributes: they are parsecraft's settings now, so a
    change in ``parsecraft.toml`` has to show up here or the parse cache
    would serve documents the new routing never produced.
    """
    routing = settings.routing
    parts = [
        f"backends:{','.join(sorted(environment.backends))}",
        f"extras:{','.join(sorted(environment.installed_extras))}",
    ]
    if routing.only_backend is not None:
        parts.append(f"only:{routing.only_backend}")
    if routing.preferred_backend is not None:
        parts.append(f"preferred:{routing.preferred_backend}")
    if routing.engines:
        parts.append(f"engines:{','.join(sorted(routing.engines))}")
    if routing.allow_ocr is not None:
        parts.append(f"ocr:{'on' if routing.allow_ocr else 'off'}")
    return "|".join(parts)


def cache_identity(environment: EnvironmentInfo, settings: ParsecraftRoutingConfig, images: ImageStore) -> str:
    """Routing inputs that change this backend's output without changing its class.

    Everything that decides which backend leads belongs here, because it changes
    what the document is parsed into. Two of those are not kompend's own config:

    * the installed extra set — a user adding or removing a parsecraft extra
      changes which backends are eligible;
    * the parsecraft version — upstream changes routing without kompend changing
      anything (the NATIVE tier flip, and mineru's ``estimated_vram_gb`` moving
      the lead from mineru to docling without touching eligibility).

    The version earns its churn: a bump is rare, coincides with an upgrade, and
    there is no way to know from here whether it changed what a backend
    produces. A backend *rename* alone would not need it, but that is the only
    case, and it is indistinguishable from this one at this boundary.

    ``judge_extras:1`` is a one-time marker, not a lever: ``DeterministicJudge``
    used to be built without the installed-extra set (knox-j5ig), so routing
    changed while every config value — and this identity — stayed put. The
    marker invalidates caches that outlived the over-demoting build; it
    carries no meaning beyond that transition.

    The no-judge, balanced-default case keeps an empty *routing* part: it is the
    rule table's neutral axis, and emitting it would re-invalidate every vault
    for no behavioural change.
    """
    routing = settings.routing
    host = f"extras:{','.join(sorted(environment.installed_extras))}|parsecraft:{_parsecraft_version()}|judge_extras:1"
    # Routing segments: judge (when configured) or preference (when non-default),
    # plus independent `effort:` and `ocr:` segments when the user set them.
    # Joining with `|` keeps each lever its own slot so the cache key flips when
    # any of them changes, but stays stable for the all-default combination —
    # the leading empty join renders as `""`, not `"|"`.
    parts: list[str] = []
    judge_spec = settings.judge_spec
    if judge_spec is not None:
        parts.append(f"judge:{judge_spec}:{routing.preference}")
    elif routing.preference is not RoutingPreference.BALANCED:
        parts.append(f"preference:{routing.preference}")
    if routing.allow_ocr is not None:
        parts.append("ocr:on" if routing.allow_ocr else "ocr:off")
    if routing.engines:
        parts.append(f"engines:{','.join(sorted(routing.engines))}")
    if routing.preferred_backend is not None:
        parts.append(f"preferred_backend:{routing.preferred_backend}")
    if routing.only_backend is not None:
        parts.append(f"only_backend:{routing.only_backend}")
    if routing.backend_options:
        rendered = ",".join(
            f"{backend}:{','.join(f'{key}={value}' for key, value in sorted(options.items()))}" for backend, options in sorted(routing.backend_options.items())
        )
        parts.append(f"backend_options:{rendered}")
    # The image mode changes what an element carries (a cached file path or
    # an inline payload), so it belongs to the identity like every other
    # output-shaping input. It is emitted only when the run actually
    # extracts images: a run that extracts none produces the same document
    # as before, and must not re-invalidate a vault for that.
    if images.enabled:
        parts.append(f"images:{images.mode.value}")
    routing_part = "|".join(parts)
    return f"{host}|{routing_part}"


def _parsecraft_version() -> str:
    """The installed parsecraft version, or `"unknown"`.

    Part of the cache identity because upstream changes routing without kompend
    changing anything, and there is no way to tell from here whether that changed
    what a backend produces.
    """
    try:
        return version("parsecraft")
    except PackageNotFoundError:  # pragma: no cover — parsecraft is an extra
        return "unknown"
