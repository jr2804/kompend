"""Pack protocol and registry for KNOX contribution packs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

KNOWN_ROLES: tuple[str, ...] = (
    "compile",
    "resolve",
    "enrich",
    "query",
    "lint.factual-consistency",
    "lint.clarity-score",
    "lint.coverage-gap",
)


@dataclass
class Pack:
    """A contribution pack — a bundle of prompts, rules, and ontology.

    Packs provide domain-specific prompt templates, compilation rules,
    and page kind schemas (N9). Built-in packs are defined in Python;
    user-created packs live as directories with a ``pack.yaml`` manifest.

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
    page_kinds:
        Page kind definitions contributed by this pack (N9). Each entry
        maps kind name → dict of kind config.
    dependencies:
        List of other pack names this pack depends on.
    metadata:
        Arbitrary extra metadata.
    """

    name: str
    description: str = ""
    version: str = "1.0.0"
    roles: dict[str, str] = field(default_factory=dict)
    page_kinds: dict[str, dict[str, Any]] = field(default_factory=dict)
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

    def page_kinds_for(self, pack_names: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Aggregate page kind definitions from active packs."""
        active = self._resolve_active(pack_names)
        merged: dict[str, dict[str, Any]] = {}
        for pack in active:
            merged.update(pack.page_kinds)
        return merged

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
        """Register the built-in ``default`` pack."""
        self._packs["default"] = Pack(
            name="default",
            description="Built-in default pack with generic prompts for all pipeline roles.",
            version="1.0.0",
            roles={
                "compile": (
                    "You compile knowledge from the following source documents. "
                    "Extract key concepts, entities, and relationships. "
                    "Return structured JSON with 'nodes' (list of pages) and "
                    "'edges' (list of relationships between pages)."
                ),
                "resolve": (
                    "Extract named entities and resolve cross-references from the text below. "
                    "Return a JSON object with 'entities' (list of dicts with name, type, "
                    "and description) and 'relationships' (list of dicts with source, target, "
                    "and relation)."
                ),
                "enrich": (
                    "Given the following content, enrich it with additional context and details. "
                    "Add definitions, examples, and related concepts where appropriate."
                ),
                "query": (
                    "Answer the user's question based on the compiled wiki content below. "
                    "Cite specific pages and sources. If the wiki doesn't contain the "
                    "answer, say so clearly."
                ),
            },
        )


# Module-level singleton
_REGISTRY: PackRegistry | None = None


def get_registry() -> PackRegistry:
    """Return the global :class:`PackRegistry` singleton."""
    global _REGISTRY  # noqa: PLW0603
    if _REGISTRY is None:
        _REGISTRY = PackRegistry()
    return _REGISTRY
