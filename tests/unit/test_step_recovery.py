"""Tests for :mod:`kompend.compilation.step_recovery` — the step-store
resume layer that replaced the hand-rolled generation checkpoint.

Uses a real :class:`~pydantic_ai_harness.step_persistence.SqliteStepStore`
in ``tmp_path`` so the on-disk persistence contract is exercised, not a
stub.
"""

from __future__ import annotations

import asyncio
from importlib import import_module
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, SqliteStepStore

from kompend.compilation.schema import GenerationOutput
from kompend.compilation.step_recovery import (
    SegmentRecovery,
    extract_final_output,
    generation_conversation,
    pass_run_id,
    scan_segment,
    segment_run_prefix,
)

step_recovery = import_module("kompend.compilation.step_recovery")


class _Result(BaseModel):
    pages: list[str] = []


class TestKeyFormats:
    """The deterministic keys every resume decision hangs off."""

    def test_generation_conversation_is_deterministic_and_prefixed(self) -> None:
        deps = {"EXTRACTIONS": "abc", "MODEL": "ollama/m"}
        first = generation_conversation(deps)
        assert first == generation_conversation(dict(deps))
        assert first.startswith("gen-")
        assert len(first) == len("gen-") + 16

    def test_generation_conversation_changes_with_inputs(self) -> None:
        assert generation_conversation({"MODEL": "a"}) != generation_conversation({"MODEL": "b"})

    def test_segment_run_prefix(self) -> None:
        assert segment_run_prefix("gen-abc", 3) == "gen-abc-seg003"
        assert segment_run_prefix("gen-abc", 12) == "gen-abc-seg012"

    def test_pass_run_id(self) -> None:
        assert pass_run_id("gen-abc-seg001", 2, 3) == "gen-abc-seg001-a02p3"


class TestScanSegment:
    """Recovery probes against a real on-disk step store."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> SqliteStepStore:
        return SqliteStepStore(database=tmp_path / "steps.db")

    def test_empty_store_recovers_nothing(self, store: SqliteStepStore) -> None:
        recovery = scan_segment(store, "gen-abc", 0)
        assert recovery.snapshot is None
        assert recovery.interrupted is False
        assert recovery.attempts == 0
        assert recovery.history is None

    def test_finished_conversation_yields_the_snapshot(self, store: SqliteStepStore) -> None:
        conversation = "gen-abc"
        prefix = segment_run_prefix(conversation, 0)
        run_id = pass_run_id(prefix, 1, 1)
        _seed_run(store, run_id, conversation, _snapshot(run_id, _history('{"pages": []}')))

        recovery = scan_segment(store, conversation, 0)
        assert recovery.snapshot is not None
        assert recovery.interrupted is False
        assert recovery.attempts == 1
        assert recovery.history is not None

    def test_interrupted_latest_snapshot_is_flagged(self, store: SqliteStepStore) -> None:
        conversation = "gen-abc"
        prefix = segment_run_prefix(conversation, 1)
        run_id = pass_run_id(prefix, 1, 1)
        _seed_run(store, run_id, conversation, _snapshot(run_id, _history('{"pages": []}'), state="interrupted"))

        recovery = scan_segment(store, conversation, 1)
        assert recovery.snapshot is not None
        assert recovery.interrupted is True

    def test_newest_attempt_wins(self, store: SqliteStepStore) -> None:
        conversation = "gen-abc"
        prefix = segment_run_prefix(conversation, 0)
        first = pass_run_id(prefix, 1, 1)
        second = pass_run_id(prefix, 2, 1)
        _seed_run(store, first, conversation, _snapshot(first, _history('{"pages": ["old"]}')))
        _seed_run(store, second, conversation, _snapshot(second, _history('{"pages": ["new"]}')))

        recovery = scan_segment(store, conversation, 0)
        assert recovery.snapshot is not None
        assert recovery.snapshot.run_id == second
        assert recovery.attempts == 2

    def test_runs_of_other_segments_are_ignored(self, store: SqliteStepStore) -> None:
        conversation = "gen-abc"
        other = pass_run_id(segment_run_prefix(conversation, 5), 1, 1)
        _seed_run(store, other, conversation, _snapshot(other, _history('{"pages": []}')))

        assert scan_segment(store, conversation, 0).snapshot is None


class TestExtractFinalOutput:
    """Structured-output reconstruction from a persisted transcript."""

    def test_reads_the_final_result_tool_call(self) -> None:
        messages = _history('{"pages": ["Alpha", "Beta"]}')
        result = extract_final_output(messages, _Result)
        assert result is not None
        assert result.pages == ["Alpha", "Beta"]

    def test_last_result_wins(self) -> None:
        messages = [
            _final_response('{"pages": ["old"]}', call_id="c1"),
            _tool_return(call_id="c1"),
            _final_response('{"pages": ["new"]}', call_id="c2"),
            _tool_return(call_id="c2"),
        ]
        result = extract_final_output(messages, _Result)
        assert result is not None
        assert result.pages == ["new"]

    def test_other_tool_calls_are_ignored(self) -> None:
        fetch = ModelResponse(parts=[ToolCallPart(tool_name="fetch_concepts", args="{}", tool_call_id="f1")])
        settle = ModelRequest(parts=[ToolReturnPart(tool_name="fetch_concepts", content="[]", tool_call_id="f1")])
        assert extract_final_output([fetch, settle], _Result) is None

    def test_mismatched_shape_validates_to_the_default(self) -> None:
        # ``pages`` has a default, so an unrelated payload still validates —
        # it comes back as an EMPTY output. Callers must treat an empty
        # result as "not recovered" (the generation flow does).
        result = extract_final_output(_history('{"totally": "different"}'), _Result)
        assert result is not None
        assert result.pages == []

    def test_garbage_args_validate_to_the_default(self) -> None:
        result = extract_final_output(_history("not-json"), _Result)
        assert result is not None
        assert result.pages == []


class TestGenerationOutputExtraction:
    """The real generation schema round-trips through a transcript."""

    def test_generation_output_survives_the_round_trip(self) -> None:
        output = GenerationOutput(pages=[])
        payload = output.model_dump_json()
        result = extract_final_output(_history(payload), GenerationOutput)
        assert result == output


class TestSegmentRecovery:
    """The frozen carrier's defaults and history repair."""

    def test_history_repairs_interrupted_transcripts(self) -> None:
        messages: list[ModelMessage] = [_final_response('{"pages": []}')]
        recovery = SegmentRecovery(snapshot=_snapshot("r", messages, state="interrupted"), interrupted=True, attempts=1)
        assert recovery.history is not None

    def test_no_snapshot_means_no_history(self) -> None:
        recovery = SegmentRecovery(snapshot=None, interrupted=False, attempts=0)
        assert recovery.history is None


class TestSqliteRoundTrip:
    """A store re-opened from disk still serves the recovery probe."""

    def test_persistence_across_connections(self, tmp_path: Path) -> None:
        database = tmp_path / "steps.db"
        conversation = "gen-persist"
        prefix = segment_run_prefix(conversation, 2)
        run_id = pass_run_id(prefix, 1, 4)
        first = SqliteStepStore(database=database)
        _seed_run(first, run_id, conversation, _snapshot(run_id, _history('{"pages": ["kept"]}')))

        reopened = SqliteStepStore(database=database)
        recovery = scan_segment(reopened, conversation, 2)
        assert recovery.snapshot is not None
        assert recovery.attempts == 1
        assert recovery.interrupted is False
        recovered = extract_final_output(recovery.snapshot.messages, _Result)
        assert recovered is not None
        assert recovered.pages == ["kept"]


def _history(payload: str) -> list[ModelMessage]:
    """A settled tool-call round (call + return), the minimum sendable transcript."""
    return [_final_response(payload), _tool_return()]


def _final_response(payload: str, call_id: str = "call-1") -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name="final_result", args=payload, tool_call_id=call_id)])


def _tool_return(call_id: str = "call-1") -> ModelMessage:
    return ModelRequest(parts=[ToolReturnPart(tool_name="final_result", content="ok", tool_call_id=call_id)])


def _snapshot(run_id: str, messages: list[ModelMessage], *, state: str = "complete") -> ContinuableSnapshot:
    return ContinuableSnapshot(run_id=run_id, step_index=1, messages=messages, state=state)  # type: ignore[arg-type]


def _seed_run(store: SqliteStepStore, run_id: str, conversation: str, snapshot: ContinuableSnapshot) -> None:
    async def _seed() -> None:
        await store.register_run(RunRecord(run_id=run_id, conversation_id=conversation))
        await store.save_snapshot(snapshot)

    asyncio.run(_seed())


def test_module_exports_the_recovery_types() -> None:
    """The generation flow imports these names from this module."""
    for name in ("SegmentRecovery", "extract_final_output", "generation_conversation", "pass_run_id", "scan_segment", "segment_run_prefix"):
        assert hasattr(step_recovery, name)
