"""Tests for the agentic extraction flow's window-aware digest handling.

The AGENTIC strategy injects the running :class:`ExtractionMemory` digest
into every section request. These tests pin the window arithmetic that
keeps that request inside the configured context window on long documents:
the digest is consolidated by a model call once it exceeds the per-section
budget, provider overflow errors fail the section without a blind retry,
and nothing is ever cropped.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import kompend.compilation.flows.agentic as agentic_module
from kompend.compilation._two_phase_core import context_safety_tokens
from kompend.compilation.extraction_memory import ExtractionMemory
from kompend.compilation.flows.agentic import (
    _consolidate_digest,
    _digest_budget_tokens,
    _is_context_overflow,
    _RollingConsolidation,
    _running_digest,
)
from kompend.compilation.schema import Concept, Entity
from kompend.llm.context_window import ContextSource, ContextWindow

_WINDOW = 32768


class _StubConsolidator:
    """Records consolidation calls and returns scripted results."""

    def __init__(self, results: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._results = list(results or [])

    def run(self, deps: Any) -> str:
        self.calls.append({"previous": deps.previous, "delta": deps.delta, "target_tokens": deps.target_tokens})
        if self._results:
            return self._results.pop(0)
        return ""


# -- budget arithmetic --------------------------------------------------------


def test_digest_budget_subtracts_every_window_term() -> None:
    cw = _context_window()
    section = "x" * 4000  # ~1000 tokens at 4 chars/token
    budget = _digest_budget_tokens(cw, static_tokens=2000, section_text=section)
    assert budget == _WINDOW - cw.answer_reserve - context_safety_tokens(_WINDOW) - 2000 - 1000


def test_digest_budget_shrinks_with_the_section() -> None:
    small = _digest_budget_tokens(_context_window(), 0, "x" * 400)
    large = _digest_budget_tokens(_context_window(), 0, "x" * 40000)
    assert large < small


# -- overflow detection -------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("request (33145 tokens) exceeds the available context size (32768 tokens), try increasing it", True),
        ("Error code: 400 - {'code': 'context_length_exceeded', 'message': 'reduce the length of the messages'}", True),
        ("This model's maximum context length is 32768 tokens. However, you requested 33145 tokens", True),
        ("Error code: 404 - {'message': 'The model no longer exists'}", False),
        ("connection reset by peer", False),
    ],
)
def test_is_context_overflow_matches_provider_markers(message: str, expected: bool) -> None:
    assert _is_context_overflow(ValueError(message)) is expected


def test_is_context_overflow_walks_the_exception_chain() -> None:
    root = ValueError("request (33145 tokens) exceeds the available context size (32768 tokens)")
    wrapped = RuntimeError("agent call failed")
    wrapped.__cause__ = root
    assert _is_context_overflow(wrapped) is True


# -- digest gating ------------------------------------------------------------


def test_running_digest_injects_the_full_digest_while_it_fits() -> None:
    memory = _memory(sections=2)
    consolidator = _StubConsolidator()
    digest = _running_digest(memory, _RollingConsolidation(), budget_tokens=10_000, consolidator=consolidator, source="doc")
    assert digest == memory.summarize()
    assert consolidator.calls == []


def test_running_digest_consolidates_once_over_budget() -> None:
    memory = _memory(sections=50)
    state = _RollingConsolidation()
    consolidator = _StubConsolidator(results=["Concepts: consolidated"])
    budget = 250

    digest = _running_digest(memory, state, budget_tokens=budget, consolidator=consolidator, source="doc")

    assert digest == "Concepts: consolidated"
    assert len(consolidator.calls) == 1
    call = consolidator.calls[0]
    assert call["previous"] == ""
    assert call["target_tokens"] == budget
    # the snapshot advanced: nothing new arrives, no second call
    again = _running_digest(memory, state, budget_tokens=budget, consolidator=consolidator, source="doc")
    assert again == "Concepts: consolidated"
    assert len(consolidator.calls) == 1


def test_running_digest_without_room_runs_without_context() -> None:
    memory = _memory()
    digest = _running_digest(memory, _RollingConsolidation(), budget_tokens=0, consolidator=_StubConsolidator(), source="doc")
    assert digest == ""


def test_consolidate_digest_retries_once_with_a_tighter_target() -> None:
    memory = _memory(sections=3)
    state = _RollingConsolidation()
    oversized = "y" * 4_000  # ~1000 tokens, over a 500-token budget
    consolidator = _StubConsolidator(results=[oversized, "fits"])

    result = _consolidate_digest(consolidator, memory, state, budget_tokens=500, source="doc")

    assert result == "fits"
    assert [call["target_tokens"] for call in consolidator.calls] == [500, 250]


def test_consolidate_digest_burns_when_the_result_stays_over_budget() -> None:
    memory = _memory(sections=3)
    oversized = "y" * 4_000
    consolidator = _StubConsolidator(results=[oversized, oversized])

    with pytest.raises(RuntimeError, match="cannot be consolidated"):
        _consolidate_digest(consolidator, memory, _RollingConsolidation(), budget_tokens=500, source="doc")


def _memory(sections: int = 1) -> ExtractionMemory:
    memory = ExtractionMemory()
    for idx in range(sections):
        memory.ingest(
            SimpleNamespace(
                concepts=[Concept(name=f"concept-{idx}", definition=f"definition {idx}")],
                entities=[Entity(name=f"entity-{idx}", type="thing")],
                tags=[f"tag-{idx}"],
                summary=f"section {idx} summary",
            )
        )
    return memory


def test_context_overflow_fails_the_section_without_a_blind_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _OverflowAgent:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run_with_messages(self, deps: Any, *, message_history: Any = None) -> Any:
            calls.append(deps.source_id)
            raise ValueError("request (33145 tokens) exceeds the available context size (32768 tokens), try increasing it")

    class _QuietConsolidator:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, deps: Any) -> str:
            return ""

    monkeypatch.setattr(agentic_module, "ExtractionAgent", _OverflowAgent)
    monkeypatch.setattr(agentic_module, "MemoryConsolidationAgent", _QuietConsolidator)

    result = agentic_module.extract_agentic(_fake_compiler(batch_retry_attempts=3), "doc.pdf", ["section text"], "sys")

    assert result is None  # every section failed -> no extraction
    assert calls == ["doc.pdf:0"]  # one call, not 1 + batch_retries


# -- section loop -------------------------------------------------------------


def _fake_compiler(batch_retry_attempts: int) -> SimpleNamespace:
    llm = SimpleNamespace(
        batch_retry_attempts=batch_retry_attempts,
        reasoning_effort=None,
        delta_retry_enabled=True,
        context_manager=None,
    )
    return SimpleNamespace(
        _config=SimpleNamespace(no_cache=True, skills={}, llm=llm),
        _model="ollama/noop",
        _timeout=1.0,
        _context_window=_context_window(),
        _extract_system_tokens=100,
        _asset_store=None,
        _usage_tracker=None,
        _batch_tracker=SimpleNamespace(record=lambda **kwargs: None),
        _cache_store=None,
    )


def _context_window(max_input: int = _WINDOW) -> ContextWindow:
    return ContextWindow(max_input=max_input, max_output=4096, source=ContextSource.FALLBACK)
