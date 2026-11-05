"""``kompend list-parsers`` — the one ingestion parser and the formats it reads."""

from __future__ import annotations

import json
import sys

from kompend.cli.args import JsonOutput
from kompend.logging import get_console
from kompend.preprocessing.parsers import PARSER_PARSECRAFT, build_parser


def list_parsers_cmd(
    *,
    json_output: JsonOutput = False,
) -> None:
    """Show the ingestion parser and the suffixes it claims.

    There is one parser — parsecraft — which routes each page across its own
    sub-backends. The command stays because the format list it prints is the
    honest answer to "what can this vault ingest", and it differs per install
    (a parsecraft extra adds a backend, which can add a format).
    """
    parser = build_parser().parser
    extensions = sorted(str(extension) for extension in parser.supported_extensions)
    payload = [
        {
            "name": PARSER_PARSECRAFT,
            "type": "built-in",
            "extensions": extensions,
            "figure_enrichment": parser.figure_enrichment,
            "metadata": {},
        }
    ]

    if json_output:
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    get_console().print(f"Ingestion parser: {PARSER_PARSECRAFT} (built-in)")
    get_console().print(f"Formats ({len(extensions)}): {', '.join(extensions)}", markup=False)
    if parser.figure_enrichment:
        get_console().print("Figures: extracted when the winning backend emits them", markup=False)
