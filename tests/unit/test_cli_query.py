"""Tests for `knox query` output-format encoding.

Covers the four LlmDataFormat outputs (text/json/toon/ison). Encoding logic
is exercised directly so the test stays fast and deterministic (no LLM,
no vector store, no live CLI).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import toon_format as _toon

from knox.cli.cmds.knowledge.query import (
    encode_raw_results,
    encode_structured,
    structured_answer,
)
from knox.core.config._configs import LlmDataFormat


@dataclass
class _Synth:
    answer: str
    citations: list[dict[str, Any]]


@dataclass
class _Hit:
    title: str
    source_file: str | None
    snippet: str
    score: float


class TestStructuredAnswer:
    def test_builds_answer_and_citations(self) -> None:
        payload = structured_answer(_synth())
        assert payload["answer"] == "Neural nets approximate functions via layers."
        # wiki_link is None when no wiki is supplied (node_type lookup unavailable)
        assert payload["citations"] == [
            {"article": "Neural Networks", "wiki_link": None, "source": "src.pdf", "snippet": "A neural net...", "score": 0.9},
        ]


class TestEncodeStructured:
    def test_json_is_pretty_parsed(self) -> None:
        out = encode_structured(structured_answer(_synth()), LlmDataFormat.JSON)
        parsed = json.loads(out)
        assert parsed["answer"].startswith("Neural nets")
        # pretty-printed (indent) — multi-line
        assert "\n" in out

    def test_toon_round_trips(self) -> None:
        payload = structured_answer(_synth())
        out = encode_structured(payload, LlmDataFormat.TOON)
        assert out  # non-empty
        parsed = _toon.decode(out)
        assert isinstance(parsed, dict)
        assert parsed["answer"] == payload["answer"]

    def test_ison_round_trips(self) -> None:
        payload = structured_answer(_synth())
        out = encode_structured(payload, LlmDataFormat.ISON)
        assert out  # non-empty
        # ISON wraps the answer in a single-row table so the scalar survives
        assert payload["answer"] in out
        assert "table.answer" in out
        assert "table.citations" in out

    def test_toon_and_ison_are_more_compact_than_json(self) -> None:
        """Token-efficient formats must not be larger than pretty JSON."""
        payload = structured_answer(_synth())
        json_len = len(encode_structured(payload, LlmDataFormat.JSON))
        assert len(encode_structured(payload, LlmDataFormat.TOON)) <= json_len
        assert len(encode_structured(payload, LlmDataFormat.ISON)) <= json_len


class TestEncodeRawResults:
    def test_json_raw_results_parse(self) -> None:
        results = [_Hit("Title", "f.pdf", "snip", 0.5)]
        parsed = json.loads(encode_raw_results(results, LlmDataFormat.JSON))
        assert parsed[0]["article"] == "Title"
        assert parsed[0]["score"] == 0.5

    def test_toon_raw_results_decode(self) -> None:
        results = [_Hit("Title", "f.pdf", "snip", 0.5)]
        out = encode_raw_results(results, LlmDataFormat.TOON)
        assert "Title" in out


def _synth() -> _Synth:
    return _Synth(
        answer="Neural nets approximate functions via layers.",
        citations=[
            {"article": "Neural Networks", "article_id": "neural-networks", "source": "src.pdf", "snippet": "A neural net...", "score": 0.9},
        ],
    )
