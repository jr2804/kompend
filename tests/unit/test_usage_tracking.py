"""Tests for the throughput tracker (tok/s) and provider connection limits."""

from __future__ import annotations

import threading
import time
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
        t = UsageTracker()
        t.record(prompt_tokens=10, completion_tokens=500, model="m")
        time.sleep(0.2)
        t.record(prompt_tokens=10, completion_tokens=1500, model="m")
        overall = t.overall_rate()
        assert 2000 / 0.6 < overall < 2000 / 0.1  # sane magnitude; timing-dependent
        assert t.completion_tokens == 2000
        assert t.call_count == 2

    def test_window_limits_events(self) -> None:
        t = UsageTracker(throughput_window_tokens=100)
        # First call: 500 tokens, long gap. Then 5 fast calls of 100 tokens.
        t.record(prompt_tokens=5, completion_tokens=500, model="m")
        time.sleep(0.4)
        for _ in range(5):
            t.record(prompt_tokens=5, completion_tokens=100, model="m")
            time.sleep(0.01)
        # Window 100 -> only the last 100-token call(s) inside the window,
        # which completed within the fast cluster -> recent rate must be
        # much higher than the overall rate (which spans the 0.4s gap).
        recent = t.recent_rate()
        overall = t.overall_rate()
        assert recent > 0.0
        assert recent > overall

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
