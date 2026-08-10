"""Type aliases for common patterns to reduce typing.Any usage."""

from typing import Any, TypeVar

from jinja2 import ChoiceLoader, Environment

# JSON-related types
JSON = Any  # Unparsed JSON string or parsed JSON structure (dict/list/primitives)

TypeT = TypeVar("TypeT")
TypeType = type[TypeT] | None  # Type object or None
JSONDict = dict[str, JSON]  # JSON object (dictionary)
JSONList = list[JSON]  # JSON array (list)
JSONValue = JSON  # Any JSON value (string, number, boolean, null, object, array)

# Wiki and page types
Page = JSONDict  # Wiki page structure
Pages = list[Page]  # List of wiki pages

# Metadata types
MetadataValue = str | bool | int | float | list["MetadataValue"] | dict[str, "MetadataValue"]  # Value stored in WikiNode.metadata / provenance dicts

# Configuration types
ConfigDict = dict[str, JSON]  # Configuration dictionary
ConfigValue = JSON  # Configuration value (can be any JSON type)

# Cache types
CacheValue = JSON  # Cached value (can be any JSON-serializable type)

# Template types
TemplateContext = dict[str, JSON]  # Template rendering context
Jinja2Env = Environment  # Jinja2 template environment
Jinja2Loader = ChoiceLoader  # Jinja2 template loader

# Agent types
AgentResult = JSON  # Result from agent execution (can be any structure)

# Parser types
ParsedResult = JSON  # Result from document parsing (can be any structure)

# External library types
try:
    import zvec

    ZvecCollection = zvec.Collection
    ZvecDoc = zvec.Doc
except ImportError:
    ZvecCollection = Any  # Type alias for zvec.Collection object
    ZvecDoc = Any  # Type alias for zvec.Doc object
