"""MCP list_parsers tool - the ingestion parser and the suffixes it claims."""

from __future__ import annotations

from typing import Any

from kompend.mcp.tools._helpers import ToolStatus
from kompend.preprocessing.parsers import PARSER_PARSECRAFT, build_parser


def list_parsers(
    json_output: bool = False,
) -> dict[str, Any]:
    """Show the ingestion parser and the suffixes it claims."""
    parser = build_parser().parser
    extensions = sorted(str(extension) for extension in parser.supported_extensions)
    parsers = [
        {
            "name": PARSER_PARSECRAFT,
            "type": "built-in",
            "extensions": extensions,
            "figure_enrichment": parser.figure_enrichment,
            "metadata": {},
        }
    ]

    if json_output:
        return {"status": ToolStatus.OK, "parsers": parsers}

    return {"status": ToolStatus.OK, "parsers_text": f"Ingestion parser: {PARSER_PARSECRAFT} (built-in)\nFormats ({len(extensions)}): {', '.join(extensions)}"}
