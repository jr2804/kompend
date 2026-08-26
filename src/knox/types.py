"""Shared structural types for KNOX.

The JSON types are precise recursive unions — NOT ``Any`` aliases. Data
read from LLM output, config files, caches, and stores is JSON-shaped and
unknown at compile time; these unions let ``ty`` force every consumer to
narrow before use instead of silently propagating ``Any``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

from jinja2 import ChoiceLoader, Environment
from pydantic import BaseModel

# ── JSON types ────────────────────────────────────────────────────────
# Recursive JSON value: every JSON payload in KNOX narrows to this.
# (3.11-compatible spelling; PEP 695 ``type`` statements need 3.12)
JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]

JSON = JSONValue  # Unparsed-structure values flowing through agents/stores

TypeT = TypeVar("TypeT")
TypeType = type[TypeT] | None  # Type object or None

JSONDict = dict[str, JSONValue]  # JSON object (dictionary)
JSONList = list[JSONValue]  # JSON array (list)

# Wiki page aliases were deleted: use WikiNode / ParsedDocument directly.

# Metadata types
MetadataValue = str | bool | int | float | list["MetadataValue"] | dict[str, "MetadataValue"]  # Value stored in WikiNode.metadata / provenance dicts

# Configuration types
ConfigDict = dict[str, JSONValue]  # Configuration dictionary
ConfigValue = JSONValue  # Configuration value (a concrete JSON type)
Attributes = dict[str, str]  # Source-level key=value pairs (labels)


# Cache types
# Decoded cache values: JSON round-trip plus the tagged runtime types
# reconstructed by knox.core.cache.store.decode_json.
@runtime_checkable
class DataclassInstance(Protocol):
    """Structural type for any ``@dataclass`` instance."""

    __dataclass_fields__: ClassVar[dict[str, Any]]


CacheValue = JSONValue | Path | bytes | tuple["CacheValue", ...] | DataclassInstance

# Template types
TemplateContext = dict[str, JSONValue]  # Template rendering context
Jinja2Env = Environment  # Jinja2 template environment
Jinja2Loader = ChoiceLoader  # Jinja2 template loader

# Agent types
AgentResult = BaseModel | str  # Structured pydantic output or raw text response

# Parser results are ParsedDocument (see knox.core.models) — no alias needed.

# External library types
ZvecCollection = Any  # ponytail: optional-dep type unavailable when zvec missing; revisit if zvec ships stubs
ZvecDoc = Any
