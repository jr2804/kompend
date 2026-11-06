"""parsecraft's routing configuration, read through parsecraft's own engine.

kompend no longer mirrors parsecraft's routing vocabulary (preference, OCR
permission, engine list, backend pins, per-backend options): those are
parsecraft's settings and they live in parsecraft's files. This module is the
single place where kompend asks parsecraft what routing it should run, so the
pipeline, the CLI and the MCP server cannot drift apart.

Two rules shape it:

* **Configured values win; host facts fill in only what nobody configured.**
  Which external engines exist and how much VRAM is usable are properties of the
  machine, not settings anyone writes. They are supplied when the constraints
  are built, and only for keys no config layer set — so an explicit
  ``routing.engines = []`` still means "no engines". The loaded settings
  themselves stay exactly what the files say, which is what the parse-cache
  identity is derived from: a probe result must not look like a configured
  routing change, or every vault would re-parse whenever the local machine
  happens to gain a tool.
* **kompend's one policy value stays a default, not a fork.** A wiki build is
  best-effort extraction, so the fallback depth is 3 (``max_passes``) rather
  than parsecraft's fail-loud 1. It is injected at the default layer, so a user
  who wants the upstream behaviour writes ``max_passes = 1``.

The routing judge is parsecraft's setting too. ``[judge] model`` in one of the
files above names it, and ``PARSECRAFT_JUDGE_MODEL`` overrides it for one run;
kompend mirrors neither the provider vocabulary nor the model token. It does not
have to: the loader below is always paired with
:func:`parsecraft.cli.config.config_scope`, which installs the same settings for
the judge, the provider endpoints and the request timeout to read. That pairing
is what lets a ``kompend`` run honour ``<vault>/.kompend/parsecraft.toml``
exactly as it honours every other routing key.

When kompend grows a decision model of its own (``[decision]``), it will reach
parsecraft through ``register_judge_provider``: a loader that answers
``RoutingJudge.rank`` from the kompend decision model, addressed by
``[judge] model = "kompend/<name>"``. That keeps the direction of dependency
one-way — parsecraft never imports kompend — and means no kompend key has to
name a provider.

``allow_ocr`` is where the two rules meet. Left unset it is ``None``, which reads
as "let the eligibility derivation decide" — but ``RoutingSettings``'
``routing_constraints`` then omits the key entirely and ``RoutingConstraints``'
own default (``True``) applies. The derivation the settings meant is
``bool(environment.ocr_backends)``, which the probe already answers, so it is
supplied as a host fact: a machine with no OCR backend would otherwise be told
OCR is permitted. An explicit ``allow_ocr`` still wins.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from parsecraft.cli.config import ParsecraftConfig, RoutingSettings, build_engine, config_scope
from parsecraft.config import ConfigLayer
from parsecraft.environment import EnvironmentInfo
from parsecraft.providers._endpoints import select_judge_spec
from parsecraft.routing import RoutingConstraints

#: kompend's fallback depth. parsecraft's default of 1 means one attempt with no
#: fallback entry, so a backend that is missing, broken or excluded by the GPU
#: gate fails the document; a wiki build prefers a degraded parse to a failed
#: one. Injected at the default layer — a user file overrides it.
KOMPEND_MAX_PASSES: Final[int] = 3

#: ``[routing]`` keys that describe the host rather than a choice. Everything a
#: user writes for them is honoured; the rest is answered by the probe.
_HOST_FACT_KEYS: Final[tuple[str, ...]] = ("engines", "vram_budget_gb", "gpu_usable", "allow_ocr")

#: Every ``[routing]`` key, used to treat a settings object built in-process as
#: fully specified (its caller chose every value, including the defaults).
_ROUTING_KEYS: Final[tuple[str, ...]] = tuple(RoutingSettings.model_fields)


@dataclass(frozen=True)
class ParsecraftRoutingConfig:
    """parsecraft's loaded configuration plus which routing keys it specified.

    ``written`` is what lets a host fact fill a blank without overriding a
    deliberate value: it names the ``routing.*`` keys that came from a config
    layer (global file, vault file, project file, ``PARSECRAFT_CONFIG_*``, CLI)
    rather than from the schema defaults.
    """

    settings: ParsecraftConfig
    written: frozenset[str]

    @property
    def routing(self) -> RoutingSettings:
        return self.settings.routing

    @property
    def offline(self) -> bool:
        return self.settings.offline

    def was_written(self, key: str) -> bool:
        """Whether a config layer supplied ``routing.<key>``."""
        return key in self.written

    @property
    def judge_spec(self) -> str | None:
        """The routing judge this run should use, or ``None`` for the rule table.

        Asked of parsecraft, not re-derived: ``PARSECRAFT_JUDGE_MODEL`` first,
        then ``[judge] model``. ``select_judge_spec`` reads the *effective*
        configuration, so the settings this object holds are installed for the
        duration of the question — otherwise the answer would come from the
        ambient layer set (the global file and the cwd project file) and a
        vault's ``.kompend/parsecraft.toml`` would name a judge nothing ever
        resolves. ``config_scope`` is thread-local, so this cannot disturb a
        parse running on another worker.
        """
        with config_scope(self.settings):
            return select_judge_spec(None)

    @classmethod
    def from_settings(cls, settings: ParsecraftConfig) -> ParsecraftRoutingConfig:
        """Wrap a settings object whose caller specified every value itself.

        That is the embedding-host seam: a caller that already resolved
        parsecraft's config hands one object to every parser it builds, so
        nothing is treated as unwritten and no probe result is substituted.
        """
        return cls(settings=settings, written=frozenset(_ROUTING_KEYS))


def load_parsecraft_config(config_files: Sequence[Path]) -> ParsecraftRoutingConfig:
    """Read parsecraft's layered config with kompend's policy as the default layer.

    *config_files* are the project-layer files in priority order (lowest
    first): the vault's ``.kompend/parsecraft.toml``, then ``./parsecraft.toml``.
    parsecraft itself adds the global file below them and the environment above
    them, so the full order is the documented one:

    1. schema defaults, plus kompend's fallback-depth policy below
    2. global ``~/.config/parsecraft/config.toml``
    3. vault ``<vault>/.kompend/parsecraft.toml``
    4. project ``./parsecraft.toml``
    5. env ``PARSECRAFT_CONFIG_*``

    Missing files are skipped silently by parsecraft's engine; a backend or
    engine name that does not exist raises :class:`parsecraft.config.ConfigError`
    naming the key and the accepted set.
    """
    engine = build_engine(list(config_files))
    engine.set_defaults({"routing": {"max_passes": KOMPEND_MAX_PASSES}})
    settings = engine.load()
    written = frozenset(key for key in _ROUTING_KEYS if (source := engine.source_of(f"routing.{key}")) is not None and source.layer is not ConfigLayer.DEFAULT)
    return ParsecraftRoutingConfig(settings=settings, written=written)


def constraints_for(
    config: ParsecraftRoutingConfig,
    environment: EnvironmentInfo,
    *,
    formats: set[str],
    cached_assets: frozenset[str] = frozenset(),
) -> RoutingConstraints:
    """Map the loaded config onto the constraints parsecraft's planner consumes.

    ``formats``, ``installed_extras``, ``cached_assets`` and ``offline`` are
    per-call facts, not configuration, so they are supplied here rather than
    read from the settings. The host-fact keys are resolved one by one, because
    two of them change eligibility in ways silence would hide:

    * ``engines`` gates every backend that requires an external tool
      (``pandoc``, ``tesseract``), so an unwritten key must carry the probe's
      set; an explicitly empty list stays empty.
    * ``allow_ocr`` is ``None`` when unwritten and ``routing_constraints``
      omits it rather than deriving it, leaving ``RoutingConstraints``' own
      ``True``.
    """
    routing = config.routing
    engines = routing.engines if config.was_written("engines") else sorted(environment.engines)
    allow_ocr = routing.allow_ocr if config.was_written("allow_ocr") else bool(environment.ocr_backends)
    return routing.model_copy(
        update={
            "engines": engines,
            "allow_ocr": allow_ocr,
            "vram_budget_gb": routing.vram_budget_gb if config.was_written("vram_budget_gb") else environment.vram_budget_gb,
            "gpu_usable": routing.gpu_usable if config.was_written("gpu_usable") else environment.gpu_usable,
        },
    ).routing_constraints(
        installed_extras=set(environment.installed_extras),
        formats=formats,
        cached_assets=cached_assets,
        offline=config.offline,
    )
