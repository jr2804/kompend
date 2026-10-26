"""Generation recovery over the harness step store.

Replaces the retired hand-rolled generation checkpoint
(``kompend.compilation._checkpoints``): instead of re-serializing the
produced node list after every segment, the generation agent's own
conversation is persisted by the harness :class:`StepPersistence`
capability (``AgenticHarness.step_store``), and this module reads those
transcripts back:

- A segment whose latest snapshot is *complete* carries its
  ``final_result`` tool call - the produced pages are extracted from the
  transcript and the segment is skipped without an LLM call.
- A segment whose latest snapshot is *interrupted* (process died
  mid-conversation) is resumed from its message history: the next pass
  continues the existing conversation instead of starting over.

The store keeps every attempt; run ids are single-shot per store, so a
recovered segment continues under a fresh attempt suffix while staying
inside the same conversation (``generation_conversation``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, repair_messages
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, StepStore

from kompend.core.cache.store import make_cache_key

#: pydantic-ai's implicit structured-output tool name
#: (``pydantic_ai._output.DEFAULT_OUTPUT_TOOL_NAME`` — not exported publicly).
_FINAL_RESULT_TOOL: str = "final_result"

ResultT = TypeVar("ResultT", bound=BaseModel)


@dataclass(frozen=True)
class SegmentRecovery:
    """What the step store knows about one segment before it runs.

    ``snapshot`` is the newest continuable snapshot across all attempts;
    ``interrupted`` marks a rescue point (a crash mid-tool-cycle) as
    opposed to a settled complete history.
    """

    snapshot: ContinuableSnapshot | None
    interrupted: bool
    attempts: int

    @property
    def history(self) -> list[ModelMessage] | None:
        """Messages to continue from, pairing-repaired; ``None`` = fresh run."""
        if self.snapshot is None:
            return None
        return repair_messages(list(self.snapshot.messages))


def generation_conversation(deps: Mapping[str, str]) -> str:
    """The conversation id shared by every segment of one generation run.

    Derived from the generation cache deps, so a re-run with unchanged
    inputs (model, extractions, chunk sizes, pinned facts) lands in the
    same conversation and can recover, while any input change starts a
    fresh conversation exactly like the old cache-key change did.
    """
    return f"gen-{make_cache_key(dict(deps))[:16]}"


def pass_run_id(prefix: str, attempt: int, passes: int) -> str:
    """A single-shot run id for one pass of one segment."""
    return f"{prefix}-a{attempt:02d}p{passes}"


def scan_segment(store: StepStore, conversation: str, seg_idx: int) -> SegmentRecovery:
    """Recover the newest state of one segment from the step store."""
    prefix = segment_run_prefix(conversation, seg_idx)
    runs = asyncio.run(store.list_runs(conversation_id=conversation))
    mine = [r for r in runs if r.run_id.startswith(prefix)]
    if not mine:
        return SegmentRecovery(snapshot=None, interrupted=False, attempts=0)
    # newest attempt last (list_runs is started_at-ascending)
    for run in reversed(mine):
        for include_interrupted in (False, True):
            snapshot = asyncio.run(store.latest_snapshot(run_id=run.run_id, include_interrupted=include_interrupted))
            if snapshot is not None:
                return SegmentRecovery(snapshot=snapshot, interrupted=include_interrupted, attempts=len(mine))
    return SegmentRecovery(snapshot=None, interrupted=False, attempts=len(mine))


def segment_run_prefix(conversation: str, seg_idx: int) -> str:
    """Run-id prefix shared by every attempt of one segment."""
    return f"{conversation}-seg{seg_idx:03d}"


def extract_final_output[ResultT: BaseModel](messages: Sequence[ModelMessage], result_type: type[ResultT]) -> ResultT | None:
    """Rebuild the structured output from a completed conversation.

    Scans backwards for the last ``final_result`` tool call and validates
    its payload. Returns ``None`` when the transcript carries none (an
    interrupted segment) or when the payload no longer matches
    ``result_type`` (schema drifted between attempts) - the caller then
    resumes with an LLM turn instead of trusting the transcript.
    """
    for message in reversed(messages):
        if not isinstance(message, ModelResponse):
            continue
        for part in reversed(message.parts):
            if isinstance(part, ToolCallPart) and part.tool_name == _FINAL_RESULT_TOOL:
                try:
                    return result_type.model_validate(part.args_as_dict())
                except ValidationError:
                    # The transcript predates a schema change — the caller
                    # resumes the conversation with an LLM turn instead.
                    return None
    return None
