"""Value vocabularies shared by the config schema and the CLI argument layer.

These StrEnums live here rather than in ``_configs.py`` because the CLI flag
types import them: ``kompend --help`` and ``kompend --version`` must not pay
for pydantic and the config engine, which cost most of a cold start. This
module imports nothing outside the standard library, so a flag type can name
its value domain without loading the schema.

``_configs.py`` imports every member from here and uses it in the field that
declares it, so the schema and the CLI flags can never drift apart. Use the
``kompend.core.config`` re-export outside these two boundaries.
"""

from __future__ import annotations

from enum import StrEnum


class ExportMode(StrEnum):
    """Where extracted data goes in a ``sources export``.

    - ``OFF``: skip entirely
    - ``INLINE``: markdown section in the exported file
    - ``ASSET``: JSON file in the export's assets directory
    """

    OFF = "off"
    INLINE = "inline"
    ASSET = "asset"


class LintOutput(StrEnum):
    """Volume of a lint report on the terminal."""

    QUIET = "quiet"
    SUMMARY = "summary"
    FULL = "full"


class LlmDataFormat(StrEnum):
    """Single source of truth for structured data serialization across all LLM communication.

    Covers two distinct use cases that share the same compact formats:

    1. **LLM input** — structured data serialized before injection into LLM prompts
       (e.g. concept batches in generation, probes in WICER). These use TOON/ISON
       to reduce prompt token count by 30-70%, leaving more context window for
       model output and reducing truncation-related malformed responses from
       small local models.

    2. **Downstream LLM ingestion** — structured data serialized by one agent and
       consumed by another (e.g. query command output fed to an agent consumer).
       Even though this is technically "output", the structured encoding is
       the *input* for subsequent LLM calls, so it benefits from the same
       token efficiency.

    Formats
    -------
    - ``JSON``: Standard JSON (default). Uses API-level ``response_format``
      guarantee. Always the LLM *output* format — providers enforce valid JSON
      at the protocol level via ``response_format: {"type": "json_object"}``.
      TOON/ISON are not yet widely understood by models and are counter-
      productive as output formats.
    - ``TOON``: Token-Oriented Object Notation. 30-60% fewer tokens vs JSON.
    - ``ISON``: Interchange Simple Object Notation. 30-70% fewer tokens vs JSON.
    - ``MARKDOWN``: Human-readable prose (formerly ``TEXT``). Alias for
      ``"text"`` — used when the consumer is a human reader, not an LLM.

    Note: ``response_format`` (JSON-enforced output) is orthogonal to this
    enum. The API-level ``response_format`` always produces JSON regardless
    of which ``LlmDataFormat`` is configured for prompt input. This enum
    controls *serialization of structured data before injection*, not the
    format of the LLM's response.
    """

    JSON = "json"
    TOON = "toon"
    ISON = "ison"
    MARKDOWN = "text"

    # Legacy alias — kept for backward compat with old "TEXT" references
    TEXT = "text"


class Verbosity(StrEnum):
    """Synthesis verbosity for the ``query`` command.

    Drives both the synthesis system prompt wording and the
    ``max_tokens`` budget for the synthesis LLM call. Defaults to
    ``standard``; configurable via ``search.synthesis.verbosity``.
    """

    BRIEF = "brief"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
