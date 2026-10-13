"""Pack protocol and registry for KOMPEND contribution packs."""

from __future__ import annotations

import importlib.resources as ir
import tomllib
from dataclasses import dataclass, field
from typing import Any

from kompend.logging import get_logger

KNOWN_ROLES: tuple[str, ...] = (
    "compile",
    "resolve",
    "enrich",
    "query",
    "lint.factual-consistency",
    "lint.clarity-score",
    "lint.coverage-gap",
)


# Module-level singleton
_REGISTRY: PackRegistry | None = None

logger = get_logger(__name__)


@dataclass
class Pack:
    """A contribution pack — a bundle of prompts, rules, and ontology.

    Packs provide domain-specific prompt templates, compilation rules,
    and page kind schemas (N9). Built-in packs are loaded from
    ``pack.toml`` manifests; user-created packs live as directories
    with a ``pack.toml`` manifest.

    Attributes:
    ----------
    name:
        Machine-readable pack name, e.g. ``"academic"``.
    description:
        Human-readable description.
    version:
        Semantic version string, e.g. ``"1.0.0"``.
    roles:
        Maps pipeline role → prompt template string. Roles not listed
        fall through to the default pack's prompts.
    dependencies:
        List of other pack names this pack depends on.
    metadata:
        Arbitrary extra metadata.
    """

    name: str
    description: str = ""
    version: str = "1.0.0"
    roles: dict[str, str] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PackRegistry:
    """Registry of installed contribution packs.

    The ``"default"`` pack is always registered and active. Additional
    packs can be registered via :meth:`register` or loaded from the
    :meth:`active_packs` list in the resolved config.
    """

    def __init__(self) -> None:
        self._packs: dict[str, Pack] = {}
        self._register_default()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, pack: Pack) -> None:
        """Register a pack, replacing any existing pack with the same name."""
        self._packs[pack.name] = pack
        logger.debug("Registered pack '%s' v%s", pack.name, pack.version)

    def get(self, name: str) -> Pack | None:
        """Look up a pack by name."""
        return self._packs.get(name)

    @property
    def names(self) -> list[str]:
        """Sorted list of registered pack names."""
        return sorted(self._packs)

    @property
    def all(self) -> list[Pack]:
        """All registered packs."""
        return list(self._packs.values())

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def prompt_for(self, role: str, pack_names: list[str] | None = None) -> str | None:
        """Return the prompt template for *role* across active packs.

        Packs are searched in order. The first pack that defines a
        prompt for *role* wins.
        """
        active = self._resolve_active(pack_names)
        for pack in active:
            if role in pack.roles:
                return pack.roles[role]
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_active(self, pack_names: list[str] | None) -> list[Pack]:
        """Return the ordered list of active packs, with defaults last."""
        names = list(pack_names) if pack_names else []
        if "default" not in names:
            names.insert(0, "default")

        resolved: list[Pack] = []
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            pack = self._packs.get(name)
            if pack is None:
                logger.warning("Pack '%s' not found — skipping", name)
                continue
            resolved.append(pack)
        return resolved

    def _register_default(self) -> None:
        """Register the built-in ``default`` pack loaded from TOML."""
        self._packs["default"] = load_pack_toml("kompend.packs.builtins.default")


def load_pack_toml(package: str, filename: str = "pack.toml") -> Pack:
    """Load a :class:`Pack` from a TOML manifest inside a package.

    Parameters
    ----------
    package:
        Dotted package path, e.g. ``"kompend.packs.builtins.default"``.
    filename:
        Name of the TOML file inside the package directory.
    """
    ref = ir.files(package) / filename
    data = tomllib.loads(ref.read_text(encoding="utf-8"))

    pack_section = data.get("pack", {})
    roles_section = data.get("roles", {})
    roles = {role: cfg["prompt"] for role, cfg in roles_section.items() if isinstance(cfg, dict) and "prompt" in cfg}

    return Pack(
        name=pack_section.get("name", ""),
        description=pack_section.get("description", ""),  # nocheck: pack TOML schema key
        version=pack_section.get("version", "1.0.0"),
        roles=roles,
        dependencies=list(pack_section.get("dependencies", [])),
        metadata={},
    )


def get_registry() -> PackRegistry:
    """Return the global :class:`PackRegistry` singleton."""
    global _REGISTRY  # noqa: PLW0603
    if _REGISTRY is None:
        _REGISTRY = PackRegistry()
    return _REGISTRY
