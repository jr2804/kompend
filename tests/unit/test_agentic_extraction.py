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
from kompend.agents.deps import MemoryConsolidationDeps
from kompend.compilation._two_phase_core import Extraction, context_safety_tokens
from kompend.compilation.extraction_memory import ExtractionMemory
from kompend.compilation.flows.agentic import (
    _consolidate_digest,
    _digest_budget_tokens,
    _is_context_overflow,
    _RollingConsolidation,
    _running_digest,
)
from kompend.compilation.schema import Concept, Entity, ExtractionOutput
from kompend.llm.context_window import ContextSource, ContextWindow
from kompend.llm.request_budget import EffectiveWindow

_WINDOW = 32768


class _StubConsolidator:
    """Records consolidation calls and returns scripted results."""

    def __init__(self, results: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._results = list(results or [])

    def run(self, deps: MemoryConsolidationDeps) -> str:
        self.calls.append({"previous": deps.previous, "delta": deps.delta, "target_tokens": deps.target_tokens})
        if self._results:
            return self._results.pop(0)
        return ""


# -- section loop -------------------------------------------------------------


class _FakeCompiler:
    """Just enough compiler for the section loop (window clamp + config)."""

    def __init__(self, batch_retry_attempts: int) -> None:
        llm = SimpleNamespace(
            batch_retry_attempts=batch_retry_attempts,
            reasoning_effort=None,
            delta_retry_enabled=True,
            context_manager=None,
            default_chars_per_token=4.0,
        )
        self._config = SimpleNamespace(no_cache=True, skills={}, llm=llm)
        self._model = "ollama/noop"
        self._timeout = 1.0
        self._window = EffectiveWindow(_context_window())
        self._extract_system_tokens = 100
        self._asset_store = None
        self._usage_tracker = None
        self._batch_tracker = SimpleNamespace(record=lambda **kwargs: None)
        self._cache_store = None

    @property
    def _context_window(self) -> ContextWindow:
        return self._window.current

    def _clamp_window(self, max_input: int, *, reason: str) -> bool:
        return self._window.clamp(max_input, reason=reason)


# -- budget arithmetic --------------------------------------------------------


def test_digest_budget_subtracts_every_window_term() -> None:
    cw = _context_window()
    section = "x" * 4000  # ~1000 tokens at 4 chars/token
    budget = _digest_budget_tokens(cw, static_tokens=2000, section_text=section, chars_per_token=4.0)
    assert budget == _WINDOW - cw.answer_reserve - context_safety_tokens(_WINDOW) - 2000 - 1000


def test_digest_budget_shrinks_with_the_section() -> None:
    small = _digest_budget_tokens(_context_window(), 0, "x" * 400, chars_per_token=4.0)
    large = _digest_budget_tokens(_context_window(), 0, "x" * 40000, chars_per_token=4.0)
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
    digest = _running_digest(memory, _RollingConsolidation(), budget_tokens=10_000, consolidator=consolidator, source="doc", chars_per_token=4.0)
    assert digest == memory.summarize()
    assert consolidator.calls == []


def test_running_digest_consolidates_once_over_budget() -> None:
    memory = _memory(sections=50)
    state = _RollingConsolidation()
    consolidator = _StubConsolidator(results=["Concepts: consolidated"])
    budget = 250

    digest = _running_digest(memory, state, budget_tokens=budget, consolidator=consolidator, source="doc", chars_per_token=4.0)

    assert digest == "Concepts: consolidated"
    assert len(consolidator.calls) == 1
    call = consolidator.calls[0]
    assert call["previous"] == ""
    assert call["target_tokens"] == budget
    # the snapshot advanced: nothing new arrives, no second call
    again = _running_digest(memory, state, budget_tokens=budget, consolidator=consolidator, source="doc", chars_per_token=4.0)
    assert again == "Concepts: consolidated"
    assert len(consolidator.calls) == 1


def test_running_digest_without_room_runs_without_context() -> None:
    memory = _memory()
    digest = _running_digest(memory, _RollingConsolidation(), budget_tokens=0, consolidator=_StubConsolidator(), source="doc", chars_per_token=4.0)
    assert digest == ""


def test_running_digest_names_the_consolidation_call() -> None:
    memory = _memory(sections=50)
    consolidator = _StubConsolidator(results=["Concepts: consolidated"])
    activity: list[str] = []

    digest = _running_digest(
        memory,
        _RollingConsolidation(),
        budget_tokens=250,
        consolidator=consolidator,
        source="doc",
        activity=activity.append,
        chars_per_token=4.0,
    )

    assert digest == "Concepts: consolidated"
    assert activity == ["consolidating context"]


def test_consolidate_digest_retries_once_with_a_tighter_target() -> None:
    memory = _memory(sections=3)
    state = _RollingConsolidation()
    oversized = "y" * 4_000  # ~1000 tokens, over a 500-token budget
    consolidator = _StubConsolidator(results=[oversized, "fits"])

    result = _consolidate_digest(consolidator, memory, state, budget_tokens=500, source="doc", chars_per_token=4.0)

    assert result == "fits"
    assert [call["target_tokens"] for call in consolidator.calls] == [500, 250]


def test_consolidate_digest_burns_when_the_result_stays_over_budget() -> None:
    memory = _memory(sections=3)
    oversized = "y" * 4_000
    consolidator = _StubConsolidator(results=[oversized, oversized])

    with pytest.raises(RuntimeError, match="cannot be consolidated"):
        _consolidate_digest(consolidator, memory, _RollingConsolidation(), budget_tokens=500, source="doc", chars_per_token=4.0)


def _memory(sections: int = 1) -> ExtractionMemory:
    memory = ExtractionMemory()
    for idx in range(sections):
        memory.ingest(
            Extraction(
                source=f"section-{idx}",
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


def test_context_overflow_splits_the_section_instead_of_skipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected section is re-planned as smaller windows, not dropped."""
    calls: list[tuple[str, str]] = []
    attempts = {"n": 0}

    class _OverflowOnceAgent:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run_with_messages(self, deps: Any, *, message_history: Any = None) -> Any:
            attempts["n"] += 1
            calls.append((deps.source_id, deps.text))
            if attempts["n"] == 1:
                raise ValueError("request (33145 tokens) exceeds the available context size (8192 tokens), try increasing it")
            return ExtractionOutput(summary=deps.text), []

    class _QuietConsolidator:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, deps: Any) -> str:
            return ""

    monkeypatch.setattr(agentic_module, "ExtractionAgent", _OverflowOnceAgent)
    monkeypatch.setattr(agentic_module, "MemoryConsolidationAgent", _QuietConsolidator)

    section = "x" * 40_000
    compiler = _fake_compiler(batch_retry_attempts=3)

    result = agentic_module.extract_agentic(compiler, "doc.pdf", [section], "sys")

    assert result is not None
    assert compiler._context_window.max_input == 8192
    # The rejected section plus the windows it was split into.
    assert len(calls) == 3
    # The retried payload is a window, never the same rejected section.
    assert all(text != section for _, text in calls[1:])
    assert "".join(text for _, text in calls[1:]) == section


def test_agents_are_rebuilt_against_the_clamped_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clamp rebuilds the agents that bake the window in at construction.

    The section agent and the consolidator turn the window into harness
    compaction thresholds and the request's output ceiling when they are
    built. A stale value would keep both sized for a window the provider has
    already rejected, so a clamp has to reach them too.
    """
    section_windows: list[int] = []
    consolidator_windows: list[int] = []
    attempts = {"n": 0}

    class _RecordingAgent:
        def __init__(self, **kwargs: Any) -> None:
            section_windows.append(kwargs["context_window"])

        def run_with_messages(self, deps: Any, *, message_history: Any = None) -> Any:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ValueError("request (33145 tokens) exceeds the available context size (8192 tokens), try increasing it")
            return ExtractionOutput(summary=deps.text), []

    class _RecordingConsolidator:
        def __init__(self, **kwargs: Any) -> None:
            consolidator_windows.append(kwargs["context_window"])

        def run(self, deps: Any) -> str:
            return ""

    monkeypatch.setattr(agentic_module, "ExtractionAgent", _RecordingAgent)
    monkeypatch.setattr(agentic_module, "MemoryConsolidationAgent", _RecordingConsolidator)

    compiler = _fake_compiler(batch_retry_attempts=3)

    result = agentic_module.extract_agentic(compiler, "doc.pdf", ["x" * 40_000], "sys")

    assert result is not None
    # Built at the configured window, rebuilt at the provider's limit.
    assert section_windows == [_WINDOW, 8192]
    assert consolidator_windows == [_WINDOW, 8192]


def _fake_compiler(batch_retry_attempts: int) -> Any:
    return _FakeCompiler(batch_retry_attempts)


def _context_window(max_input: int = _WINDOW) -> ContextWindow:
    return ContextWindow(max_input=max_input, max_output=4096, source=ContextSource.FALLBACK)
