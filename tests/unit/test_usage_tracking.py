"""Tests for the throughput tracker (tok/s) and provider connection limits."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from kompend.llm.usage import UsageTracker, tokens_per_second
from kompend.llm.user_providers import effective_workers, provider_max_connections, set_user_providers


class TestTokensPerSecond:
    def test_plain_rate(self) -> None:
        assert tokens_per_second(100, 2.0) == pytest.approx(50.0)

    def test_zero_tokens_or_time(self) -> None:
        assert tokens_per_second(0, 2.0) == 0.0
        assert tokens_per_second(100, 0.0) == 0.0


class _FakeClock:
    """Deterministic stand-in for ``time.monotonic`` — advance manually."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestUsageTracker:
    def test_empty_tracker_rates_zero(self) -> None:
        t = UsageTracker()
        assert t.overall_rate() == 0.0
        assert t.recent_rate() == 0.0
        assert t.to_dict()["overall_tok_s"] == 0.0

    def test_single_call_has_no_recent_interval(self) -> None:
        t = UsageTracker()
        t.record(prompt_tokens=10, completion_tokens=500, model="m")
        assert t.overall_rate() == 0.0  # no measurable span with one event
        assert t.recent_rate() == 0.0

    def test_overall_and_recent_rates(self) -> None:
        clock = _FakeClock()
        t = UsageTracker(_clock=clock)
        t.record(prompt_tokens=10, completion_tokens=500, model="m")
        clock.advance(2.0)
        t.record(prompt_tokens=10, completion_tokens=1500, model="m")
        # overall: 2000 tokens over the full 2.0 s span.
        assert t.overall_rate() == 1000.0
        # recent (window 1000): only the last 1500-token call, starting after
        # the first completion -> 1500 tokens over 2.0 s.
        assert t.recent_rate() == 750.0
        assert t.completion_tokens == 2000
        assert t.call_count == 2

    def test_window_limits_events(self) -> None:
        clock = _FakeClock()
        t = UsageTracker(throughput_window_tokens=100, _clock=clock)
        # First call: 500 tokens, long gap. Then 5 calls of 100 tokens each
        # 1 s apart. Events: (0.0, 500), (10..14, 100).
        t.record(prompt_tokens=5, completion_tokens=500, model="m")
        clock.advance(10.0)
        for _ in range(5):
            t.record(prompt_tokens=5, completion_tokens=100, model="m")
            clock.advance(1.0)
        # Window 100 -> only the newest 100-token call; its interval starts
        # at the previous completion (t=13) and ends at t=14 -> 100 tok/s.
        assert t.recent_rate() == 100.0
        # overall: 1000 tokens across the whole 14 s span.
        assert t.overall_rate() == pytest.approx(1000 / 14)
        assert t.recent_rate() > t.overall_rate()

    def test_record_is_thread_safe(self) -> None:
        t = UsageTracker()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(20):
                    t.record(prompt_tokens=1, completion_tokens=10, model="m")
            except Exception as exc:  # pragma: no cover - only on failure
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert not errors
        assert t.completion_tokens == 800
        assert t.call_count == 80

    def test_to_dict_includes_rates(self) -> None:
        t = UsageTracker()
        t.record(prompt_tokens=5, completion_tokens=50, model="m")
        d = t.to_dict()
        assert "overall_tok_s" in d
        assert "recent_tok_s" in d
        assert d["completion_tokens"] == 50


class TestProviderMaxConnections:
    def test_no_limit_by_default(self) -> None:
        set_user_providers({"my-llm": {"base_url": "http://x", "max_connections": None}})
        assert provider_max_connections("my-llm") is None
        assert effective_workers("my-llm/model", 4) == 4

    def test_limit_clips_workers(self) -> None:
        set_user_providers({"my-llm": {"base_url": "http://x", "max_connections": 2}})
        assert provider_max_connections("my-llm") == 2
        assert effective_workers("my-llm/model", 1) == 1  # under limit untouched
        assert effective_workers("my-llm/model", 4) == 2  # clipped

    def test_override_only_block_for_integrated_provider(self) -> None:
        # [providers.ollama] with no base_url — override-only block.
        set_user_providers({"ollama": {"max_connections": 1, "base_url": None}})
        assert provider_max_connections("ollama") == 1
        assert effective_workers("ollama/gemma", 8) == 1

    def test_unknown_provider_not_clipped(self) -> None:
        assert provider_max_connections("nope") is None
        assert effective_workers("nope/model", 5) == 5

    def test_no_slash_model_string(self) -> None:
        set_user_providers({"foo": {"max_connections": 1}})
        assert effective_workers("foo", 3) == 1

    @pytest.fixture(autouse=True)
    def _clean(self) -> Iterator[None]:
        set_user_providers({})
        yield
        set_user_providers({})
