"""N9 schema-driven compilation — page kinds and per-kind rules.

Page kinds define the structure and behaviour of wiki pages:

- ``concept`` — abstract idea with definition, background, related concepts
- ``entity`` — concrete thing with attributes, relationships
- ``comparison`` — side-by-side comparison of 2+ entities/concepts
- ``overview`` — high-level summary of a topic domain
- ``custom`` — user-defined kind with custom fields
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Built-in page kinds ────────────────────────────────────────────────

BUILTIN_KINDS: dict[str, dict[str, Any]] = {
    "concept": {
        "title": "Concept",
        "description": "An abstract idea, theory, or notion.",
        "required_fields": ["title", "tags"],
        "optional_fields": ["aliases", "defined_in", "related_concepts"],
        "compile_hints": {
            "group_by": "concept",
            "cross_source": True,
        },
    },
    "entity": {
        "title": "Entity",
        "description": "A concrete thing, person, organisation, or object.",
        "required_fields": ["title", "type"],
        "optional_fields": ["attributes", "relationships", "source"],
        "compile_hints": {
            "group_by": "entity",
            "cross_source": True,
        },
    },
    "comparison": {
        "title": "Comparison",
        "description": "Side-by-side comparison of two or more entities or concepts.",
        "required_fields": ["title", "compare"],
        "optional_fields": ["criteria", "conclusion"],
        "compile_hints": {
            "group_by": None,
            "cross_source": True,
        },
    },
    "overview": {
        "title": "Overview",
        "description": "High-level summary of a topic domain.",
        "required_fields": ["title"],
        "optional_fields": ["subtopics", "key_findings", "related_overviews"],
        "compile_hints": {
            "group_by": "domain",
            "cross_source": True,
        },
    },
    "custom": {
        "title": "Custom",
        "description": "User-defined page kind with custom frontmatter schema.",
        "required_fields": ["title", "kind"],
        "optional_fields": [],
        "compile_hints": {
            "group_by": None,
            "cross_source": False,
        },
    },
}


@dataclass
class PageKind:
    """A page kind with its schema and compilation rules.

    Attributes:
    ----------
    name:
        Machine-readable kind name (e.g. ``"concept"``).
    title:
        Human-readable title (e.g. ``"Concept"``).
    description:
        Description of what this kind represents.
    required_fields:
        Frontmatter fields that must be present.
    optional_fields:
        Frontmatter fields that may be present.
    compile_hints:
        Hints for the compilation process (e.g. cross-source merging behaviour).
    """

    name: str
    title: str = ""
    description: str = ""
    required_fields: list[str] = field(default_factory=list)
    optional_fields: list[str] = field(default_factory=list)
    compile_hints: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> PageKind:
        """Create a :class:`PageKind` from a dict (e.g. from YAML/TOML config)."""
        return cls(
            name=name,
            title=data.get("title", name.title()),
            description=data.get("description", ""),
            required_fields=list(data.get("required_fields", [])),
            optional_fields=list(data.get("optional_fields", [])),
            compile_hints=dict(data.get("compile_hints", {})),
        )


class SchemaRegistry:
    """Registry of page kinds available for compilation.

    Built-in kinds are registered automatically. Additional kinds can be
    contributed by packs (Q24) or loaded from ``schema/page_kinds.yaml``.
    """

    def __init__(self) -> None:
        self._kinds: dict[str, PageKind] = {}
        self._register_builtin()

    def register(self, kind: PageKind) -> None:
        """Register a page kind, replacing any existing kind with the same name."""
        self._kinds[kind.name] = kind

    def register_from_dict(self, name: str, data: dict[str, Any]) -> None:
        """Register a page kind from a dict (config file or pack)."""
        self.register(PageKind.from_dict(name, data))

    def get(self, name: str) -> PageKind | None:
        """Look up a page kind by name."""
        return self._kinds.get(name)

    @property
    def names(self) -> list[str]:
        """Sorted list of registered kind names."""
        return sorted(self._kinds)

    @property
    def all(self) -> list[PageKind]:
        """All registered kinds."""
        return list(self._kinds.values())

    def validate(self, kind_name: str, frontmatter: dict[str, Any]) -> list[str]:
        """Validate frontmatter against a page kind's schema.

        Returns a list of missing required fields (empty = valid).
        """
        kind = self._kinds.get(kind_name)
        if kind is None:
            return [f"Unknown page kind: {kind_name}"]
        return [f for f in kind.required_fields if f not in frontmatter]

    def _register_builtin(self) -> None:
        """Register all built-in page kinds."""
        for name, data in BUILTIN_KINDS.items():
            self._kinds[name] = PageKind.from_dict(name, data)


# Module-level singleton
_REGISTRY: SchemaRegistry | None = None


def get_schema_registry() -> SchemaRegistry:
    """Return the global :class:`SchemaRegistry` singleton."""
    global _REGISTRY  # noqa: PLW0603
    if _REGISTRY is None:
        _REGISTRY = SchemaRegistry()
    return _REGISTRY
