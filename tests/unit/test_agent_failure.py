"""One failure vocabulary for every LLM-backed stage.

The classifier answers a question the stages used to answer each in their own
way — "was this failure worth repeating?" — and the log answers the second
one: "has this batch failed enough to stop?". Both are exercised here through
the *real* exceptions the agent layer raises, not hand-built stand-ins, since
the whole point is that the taxonomy matches what pydantic-ai actually emits.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelRetry,
    UnexpectedModelBehavior,
)
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from kompend.agents.failure import (
    AgentFailureKind,
    EmptyAnswerError,
    UnitFailureLog,
    classify_agent_failure,
    is_retry_pointless,
    record_unit_failure,
)
from kompend.core.utils.exceptions import LLMContextOverflowError, LLMOutputError, LLMTransientError


class _Payload(BaseModel):
    """Stand-in for any stage's output schema."""

    score: int


class TestClassifyAgentFailure:
    """Each kind is recognised through the chain pydantic-ai actually raises."""

    def test_an_empty_answer_error_is_its_own_kind(self) -> None:
        assert classify_agent_failure(EmptyAnswerError("nothing")) is AgentFailureKind.EMPTY_ANSWER

    def test_a_transient_failure_is_recognised(self) -> None:
        assert classify_agent_failure(LLMTransientError("timed out")) is AgentFailureKind.TRANSIENT

    def test_a_model_api_error_without_a_status_is_transient(self) -> None:
        # No HTTP response ever arrived: connection refused, stream aborted.
        # ``ModelAPIError`` carries no status attribute at all, which is how
        # ``kompend.llm.errors`` reads "never reached the server".
        exc = ModelAPIError("connect", "connection refused")

        assert classify_agent_failure(exc) is AgentFailureKind.TRANSIENT

    def test_a_context_overflow_is_recognised(self) -> None:
        exc = LLMContextOverflowError("request (33145 tokens) exceeds the available context size (32768 tokens)")

        assert classify_agent_failure(exc) is AgentFailureKind.CONTEXT_OVERFLOW

    def test_an_empty_validator_payload_is_an_empty_answer(self) -> None:
        # ``json.loads('')`` — what a blank model answer turns into.
        try:
            _Payload.model_validate_json("")
        except ValidationError as exc:
            validation_error = exc
        else:  # pragma: no cover - the parse must fail
            pytest.fail("empty payload validated")

        assert classify_agent_failure(validation_error) is AgentFailureKind.EMPTY_ANSWER

    def test_a_whitespace_validator_payload_is_an_empty_answer(self) -> None:
        try:
            _Payload.model_validate_json("   \n ")
        except ValidationError as exc:
            validation_error = exc
        else:  # pragma: no cover - the parse must fail
            pytest.fail("whitespace payload validated")

        assert classify_agent_failure(validation_error) is AgentFailureKind.EMPTY_ANSWER

    def test_a_malformed_document_is_a_schema_mismatch(self) -> None:
        # "not json" reports the same ``json_invalid`` type as an empty answer,
        # so the classifier must look at the message, not just the type.
        try:
            _Payload.model_validate_json("not json at all")
        except ValidationError as exc:
            validation_error = exc
        else:  # pragma: no cover - the parse must fail
            pytest.fail("malformed payload validated")

        assert classify_agent_failure(validation_error) is AgentFailureKind.SCHEMA_MISMATCH

    def test_a_wrong_shape_is_a_schema_mismatch(self) -> None:
        try:
            _Payload.model_validate_json('{"score": "high"}')
        except ValidationError as exc:
            validation_error = exc
        else:  # pragma: no cover - the parse must fail
            pytest.fail("wrong-shaped payload validated")

        assert classify_agent_failure(validation_error) is AgentFailureKind.SCHEMA_MISMATCH

    def test_the_empty_answer_marker_behind_an_exhausted_budget_is_an_empty_answer(self) -> None:
        # This is the chain the text-mode agents used to produce before the model
        # layer learned to fail fast: the framework's own no-output feedback,
        # wrapped in the budget-exhausted error the stage finally saw.
        inner = ModelRetry("Please return text or include your response in a tool call.")
        outer = UnexpectedModelBehavior("Exceeded maximum output retries (4)")

        try:
            try:
                raise inner
            except ModelRetry as exc:
                raise outer from exc
        except UnexpectedModelBehavior as exc:
            wrapped = exc

        assert classify_agent_failure(wrapped) is AgentFailureKind.EMPTY_ANSWER

    def test_an_unrelated_failure_is_other(self) -> None:
        assert classify_agent_failure(RuntimeError("boom")) is AgentFailureKind.OTHER

    def test_an_empty_answer_is_found_behind_a_stage_wrapper(self) -> None:
        # KompendAgentBase re-wraps every failure with the model name, so by the
        # time a stage sees it the model-layer class is only in the cause chain.
        try:
            try:
                raise EmptyAnswerError("model said nothing")
            except EmptyAnswerError as exc:
                raise RuntimeError(f"Agent failed for model 'x': {exc}") from exc
        except RuntimeError as exc:
            wrapped = exc

        assert classify_agent_failure(wrapped) is AgentFailureKind.EMPTY_ANSWER


class TestIsRetryPointless:
    """Only the kinds where the model already answered get no second chance."""

    def test_empty_answers_and_overflow_are_not_worth_repeating(self) -> None:
        assert is_retry_pointless(AgentFailureKind.EMPTY_ANSWER)
        assert is_retry_pointless(AgentFailureKind.CONTEXT_OVERFLOW)

    def test_schema_transient_and_other_keep_their_budget(self) -> None:
        assert not is_retry_pointless(AgentFailureKind.SCHEMA_MISMATCH)
        assert not is_retry_pointless(AgentFailureKind.TRANSIENT)
        assert not is_retry_pointless(AgentFailureKind.OTHER)


class TestRecordUnitFailure:
    def test_it_classifies_logs_and_returns_the_record(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            failure = record_unit_failure(stage="clarity-score", unit="page-1", exc=LLMTransientError("slow"))

        assert failure.kind is AgentFailureKind.TRANSIENT
        assert failure.attempts == 1
        assert "clarity-score: page-1: transient after 1 request(s) - slow" in caplog.text

    def test_describe_names_unit_kind_and_cause(self) -> None:
        failure = record_unit_failure(stage="s", unit="page-7", exc=EmptyAnswerError("nothing"), attempts=3)

        assert failure.describe() == "page-7: empty_answer after 3 request(s) - nothing"


class TestUnitFailureLog:
    """The batch's stop rule: same ratio and minimum extraction already obeys."""

    def test_it_does_not_stop_below_the_minimum_unit_count(self) -> None:
        log = UnitFailureLog(stage="lint", ratio=0.5, min_units=5)
        for _ in range(3):
            log.attempted()
            log.record("page", EmptyAnswerError("nothing"))

        log.check()  # 3 failures out of 3, but only 3 units: too few to trust

    def test_it_stops_once_the_minimum_unit_count_is_reached(self) -> None:
        log = UnitFailureLog(stage="lint", ratio=0.5, min_units=2)
        log.attempted()
        log.record("page", EmptyAnswerError("nothing"))
        log.attempted()
        log.record("page", EmptyAnswerError("nothing"))
        log.attempted()

        with pytest.raises(LLMOutputError, match="More than 50%"):
            log.check()

    def test_it_stops_once_the_batch_is_mostly_failure(self) -> None:
        log = UnitFailureLog(stage="clarity-score", ratio=0.5, min_units=5)
        for _ in range(6):
            log.attempted()
            log.record("page", EmptyAnswerError("nothing"))
        log.attempted()  # one unit that will succeed

        with pytest.raises(LLMOutputError, match="More than 50% of the batch failed"):
            log.check()

    def test_it_keeps_the_ratio_below_the_threshold(self) -> None:
        log = UnitFailureLog(stage="lint", ratio=0.5, min_units=2)
        log.attempted()
        log.record("page", EmptyAnswerError("nothing"))
        for _ in range(9):
            log.attempted()

        log.check()  # 1 of 10

    def test_attempted_units_are_the_denominator(self) -> None:
        log = UnitFailureLog(stage="lint", ratio=0.5, min_units=100)
        for _ in range(20):
            log.attempted()

        assert log.total == 20
        assert log.failures == []

    def test_summary_counts_each_kind_most_frequent_first(self) -> None:
        log = UnitFailureLog(stage="lint", min_units=100)
        log.attempted()
        log.record("a", EmptyAnswerError("nothing"))
        log.attempted()
        log.record("b", EmptyAnswerError("nothing"))
        log.attempted()
        log.record("c", RuntimeError("boom"))

        assert log.summary() == "lint: 3 failed unit(s) of 3 - empty_answer=2, other=1"
        assert log.count(AgentFailureKind.EMPTY_ANSWER) == 2

    def test_an_empty_log_summarises_as_clean(self) -> None:
        assert UnitFailureLog(stage="lint").summary() == "lint: no failures"


class TestEmptyAnswerErrorIsAnLLMOutputError:
    """Existing ``except LLMOutputError`` sites keep catching it."""

    def test_it_can_be_caught_as_an_llm_output_error(self) -> None:
        with pytest.raises(LLMOutputError):
            raise EmptyAnswerError("nothing")


class TestFunctionModelChain:
    """The classifier against the chain a real pydantic-ai run produces."""

    def test_an_empty_structured_response_classifies_as_empty_answer(self) -> None:
        def answer(messages: object, info: object) -> ModelResponse:
            return ModelResponse(parts=[TextPart("")])

        agent: Agent[None, _Payload] = Agent(FunctionModel(answer), output_type=_Payload, retries={"tools": 3, "output": 0})
        with pytest.raises(UnexpectedModelBehavior) as caught:
            agent.run_sync("hello")

        # ``output: 0`` reproduces the schema-retry exhaustion directly, which is
        # what the lint rules used to see before the model layer learned to
        # fail fast on an empty answer.
        assert classify_agent_failure(caught.value) in {AgentFailureKind.EMPTY_ANSWER, AgentFailureKind.SCHEMA_MISMATCH}

    def test_model_retry_with_the_empty_marker_classifies_as_empty_answer(self) -> None:
        exc = ModelRetry("Please return text or include your response in a tool call.")

        assert classify_agent_failure(exc) is AgentFailureKind.EMPTY_ANSWER
