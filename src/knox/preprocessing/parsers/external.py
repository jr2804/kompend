"""External parser support for KNOX.

Two protocols:
1. **Subprocess** — external CLI tool reads source path from stdin, writes ``ParseResult`` JSON to stdout
2. **HTTP** — POST source file to a URL, receive ``ParseResult`` JSON in response

Both produce the same JSON format as ``ParsedDocument``.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import requests

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.preprocessing.parsers.base import BaseParser

from ._registry import register_parser

logger = logging.getLogger(__name__)


# ── JSON format helpers ─────────────────────────────────────────────────


def parse_result_from_json(data: dict[str, Any], source_path: Path) -> ParsedDocument:
    """Convert a JSON dict (from external parser) to a ``ParsedDocument``.

    Expected JSON format::

        {
            "title": "...",
            "author": "...",
            "language": "en",
            "page_count": 0,
            "metadata": {},
            "elements": [
                {
                    "type": "TEXT|HEADING|TABLE|LIST|FIGURE|CODE",
                    "content": "...",
                    "page": 1,
                    "reading_order": 0,
                    "heading_level": null,
                    "metadata": {},
                }
            ],
        }
    """
    elements: list[DocumentElement] = []
    for el_data in data.get("elements", []):
        try:
            raw_type = el_data.get("type", "text")
            el_type = ElementType(raw_type.lower())
        except ValueError:
            el_type = ElementType.TEXT

        elements.append(
            DocumentElement(
                type=el_type,
                content=el_data.get("content", ""),
                page=el_data.get("page"),
                reading_order=el_data.get("reading_order"),
                heading_level=el_data.get("heading_level"),
                metadata=el_data.get("metadata", {}),
            )
        )

    return ParsedDocument(
        source_path=source_path,
        elements=elements,
        title=data.get("title", source_path.stem),
        author=data.get("author", ""),
        language=data.get("language", "en"),
        page_count=data.get("page_count", 0),
        metadata=data.get("metadata", {}),
    )


# ── Subprocess parser ───────────────────────────────────────────────────


class ExternalParser(BaseParser):
    """Parser that delegates to an external CLI tool via subprocess.

    The external command receives the source file path as a command-line argument
    and writes ``ParseResult`` JSON to stdout.

    Config in ``.knox/config.yaml``::

        parsers:
          external:
            - name: myparser
              command: myparser-cli
              extensions: [".xyz"]
    """

    def __init__(
        self,
        command: str,
        extensions: list[str] | None = None,
        name: str = "",
    ) -> None:
        self._command = command
        self._exts = tuple(extensions or [])
        self._name = name or command
        super().__init__()

    @property
    def supported_extensions(self) -> set[str]:
        return set(self._exts)

    def parse(self, source_path: Path) -> ParsedDocument:
        """Run the external CLI tool and parse its JSON output."""
        logger.debug("Running external parser: %s %s", self._command, source_path)
        try:
            result = subprocess.run(  # noqa: S603
                [self._command, str(source_path)],
                capture_output=True,
                encoding="utf-8",
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"External parser '{self._command}' timed out after 60s") from None
        except FileNotFoundError:
            raise RuntimeError(f"External parser command not found: {self._command}") from None

        if result.returncode != 0:
            raise RuntimeError(f"External parser '{self._command}' exited with code {result.returncode}: {result.stderr.strip()}")

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"External parser '{self._command}' returned invalid JSON: {exc}") from exc

        return parse_result_from_json(data, source_path)

    def __repr__(self) -> str:
        return f"ExternalParser(name={self._name!r}, command={self._command!r}, exts={self._exts})"


# ── HTTP parser ─────────────────────────────────────────────────────────


class HTTPParser(BaseParser):
    """Parser that sends source files to an HTTP endpoint.

    POSTs the file to the configured URL and expects ``ParseResult`` JSON in response.

    Config in ``.knox/config.yaml``::

        parsers:
          http_url:
            endpoint: "http://localhost:8080/convert"
            extensions: [".abc"]
    """

    def __init__(
        self,
        endpoint: str,
        extensions: list[str] | None = None,
        name: str = "",
        timeout: int = 120,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._exts = tuple(extensions or [])
        self._name = name or f"http:{endpoint}"
        self._timeout = timeout

    @property
    def supported_extensions(self) -> set[str]:
        return set(self._exts)

    def parse(self, source_path: Path) -> ParsedDocument:
        """POST the source file to the HTTP endpoint and return the parsed result."""
        logger.debug("HTTP parser: POST %s -> %s", source_path, self._endpoint)
        try:
            with source_path.open("rb") as f:
                response = requests.post(
                    self._endpoint,
                    files={"file": (source_path.name, f)},
                    timeout=self._timeout,
                )
        except requests.RequestException as exc:
            raise RuntimeError(f"HTTP parser request failed for {self._endpoint}: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(f"HTTP parser returned {response.status_code} from {self._endpoint}: {response.text[:200]}")

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"HTTP parser returned invalid JSON from {self._endpoint}: {exc}") from exc

        return parse_result_from_json(data, source_path)

    def __repr__(self) -> str:
        return f"HTTPParser(name={self._name!r}, endpoint={self._endpoint!r}, exts={self._exts})"


# ── Auto-registration ───────────────────────────────────────────────────


def register_external_parsers(
    external_configs: list[dict[str, Any]] | None = None,
    http_config: dict[str, Any] | None = None,
) -> list[ExternalParser | HTTPParser]:
    """Create and register parsers from configuration dicts.

    Args:
        external_configs: List of ``{name, command, extensions}`` dicts.
        http_config: Dict with ``{endpoint, extensions}``.

    Returns:
        List of created parser instances.
    """
    parsers: list[ExternalParser | HTTPParser] = []

    if external_configs:
        for cfg in external_configs:
            parser = ExternalParser(
                command=cfg["command"],
                extensions=cfg.get("extensions"),
                name=cfg.get("name", ""),
            )
            parsers.append(parser)
            register_parser(
                name=parser._name,
                parser_type="external",
                extensions=list(parser.supported_extensions),
                metadata={"command": cfg["command"]},
            )
            logger.info("Registered external parser: %s", parser)

    if http_config:
        endpoint = http_config.get("endpoint", "")
        extensions = http_config.get("extensions")
        if endpoint:
            parser = HTTPParser(
                endpoint=endpoint,
                extensions=extensions,
            )
            parsers.append(parser)
            register_parser(
                name=parser._name,
                parser_type="http",
                extensions=list(parser.supported_extensions),
                metadata={"endpoint": endpoint},
            )
            logger.info("Registered HTTP parser: %s", parser)

    return parsers
