"""How the query command renders results.

Raw and structured output encoding, wiki links, and the captured-source
presentation.
"""

from __future__ import annotations

import json

from kompend.core.config._configs import LlmDataFormat
from kompend.core.models import CompiledWiki, ImageRef, NodeType
from kompend.core.paths import DEFAULT_OUTPUT_DIR_NAME, PROTECTED_SOURCE_SUBDIRS
from kompend.llm.format import encode_for_llm
from kompend.logging import get_console, get_logger
from kompend.output.vault import VaultWriter
from kompend.search.retriever import (
    SearchResult,
    SynthesisResult,
)
from kompend.types import JSONDict, JSONList

#: On-disk folder per node kind — mirrors ``VaultWriter._KIND_FOLDERS`` plus
#: the overview kind, which stays at the bundle root.
_KIND_FOLDERS: dict[str, str] = {**VaultWriter._KIND_FOLDERS, NodeType.OVERVIEW.value: ""}


logger = get_logger(__name__)


def is_captured_source(source_file: str | None) -> bool:
    """Return True if *source_file* is a tool-managed note in the capture lifecycle.

    Covers ``_captured/``, ``_learned/``, ``_distilled/`` and ``_archive/``
    notes — raw and archived notes are hidden from query retrieval either
    because they feed compile directly (raw) or were retired by
    ``consolidate``/``distill`` (archived).
    """
    return bool(source_file) and any(sub in source_file.replace("\\", "/") for sub in PROTECTED_SOURCE_SUBDIRS)


def structured_answer(synthesis: SynthesisResult, wiki: CompiledWiki | None = None) -> JSONDict:
    """Build the structured ``{answer, citations}`` payload for agent output."""
    node_types = node_type_index(wiki)
    return {
        "answer": synthesis.answer,
        "citations": [
            {
                "article": c["article"],
                "wiki_link": wiki_link(c.get("article_id"), node_types),
                "source": c.get("source"),
                "snippet": c["snippet"],
                "score": c["score"],
                "figures": [{"caption": str(f.get("caption", "")), "path": str(f.get("path", ""))} for f in c.get("figures", [])],
            }
            for c in synthesis.citations
        ],
    }


def node_type_index(wiki: CompiledWiki | None) -> dict[str, str]:
    """Map every wiki node id → its ``node_type`` (the wiki sub-directory)."""
    if wiki is None:
        return {}
    return {n.id: n.node_type.value for n in wiki.nodes}


def wiki_link(article_id: str | None, node_types: dict[str, str]) -> str | None:
    """Build a relative path into the compiled wiki.

    The on-disk folder mirrors the writer's kind mapping (``concept``/\u2026
    -> ``concepts/``; ``overview`` -> bundle root), not the bare node type.
    """
    if not article_id or article_id not in node_types:
        return None
    folder = _KIND_FOLDERS.get(node_types[article_id], "concepts")
    if not folder:
        return f"{DEFAULT_OUTPUT_DIR_NAME}/{article_id}.md"
    return f"{DEFAULT_OUTPUT_DIR_NAME}/{folder}/{article_id}.md"


def print_raw_results(results: list[SearchResult], format: LlmDataFormat) -> None:
    """Print raw search results to console."""
    if format == LlmDataFormat.MARKDOWN:
        for r in results:
            get_console().print(f"  [{r.score:.3f}] {r.title}")
            if r.source_file:
                get_console().print(f"        Source: {r.source_file}")
            get_console().print(f"        {r.snippet}")
            for figure in r.figures:
                get_console().print(f"        Figure: {figure_line(figure)}")
            get_console().print("")
        return

    get_console().print(encode_raw_results(results, format))


def figure_line(figure: ImageRef) -> str:
    """One line describing a figure: its caption and where the image lives."""
    caption = figure.get("caption") or ""
    path = figure.get("path") or ""
    if caption:
        return f"{caption} ({path})"
    return path


def encode_raw_results(results: list[SearchResult], fmt: LlmDataFormat) -> str:
    """Build the raw-results payload and encode it for agent consumers."""
    payload: JSONList = []
    for r in results:
        row: JSONDict = {
            "article": r.title,
            "source": r.source_file,
            "snippet": r.snippet,
            "score": round(r.score, 3),
            "figures": [{"caption": str(f.get("caption", "")), "path": str(f.get("path", ""))} for f in r.figures],
        }
        payload.append(row)
    return encode_for_llm(payload, fmt)


def encode_structured(payload: JSONDict, fmt: LlmDataFormat) -> str:
    """Encode a structured payload as json/toon/ison for agent consumers."""
    if fmt == LlmDataFormat.JSON:
        return json.dumps(payload, indent=2)
    if fmt == LlmDataFormat.ISON:
        payload = {"answer": [{"text": payload["answer"]}], "citations": payload["citations"]}
    return encode_for_llm(payload, fmt)
