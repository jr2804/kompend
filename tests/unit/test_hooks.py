"""Tests for the event-driven hooks system."""

from __future__ import annotations

from knox.core.hooks import HookEvent, clear_hooks, fire_hook, load_hooks_from_config, register_hook


class TestHookRegistry:
    def setup_method(self) -> None:
        clear_hooks()

    def teardown_method(self) -> None:
        clear_hooks()

    def test_register_and_fire(self) -> None:
        calls: list[str] = []
        register_hook(HookEvent.POST_COMPILE, lambda **kw: calls.append(HookEvent.POST_COMPILE.value))
        fire_hook(HookEvent.POST_COMPILE, wiki="fake")
        assert calls == ["post_compile"]

    def test_multiple_callbacks_same_event(self) -> None:
        results: list[int] = []
        register_hook(HookEvent.ON_CAPTURE, lambda **kw: results.append(1))
        register_hook(HookEvent.ON_CAPTURE, lambda **kw: results.append(2))
        fire_hook(HookEvent.ON_CAPTURE)
        assert results == [1, 2]

    def test_fire_no_callbacks_is_noop(self) -> None:
        fire_hook(HookEvent.PRE_COMPILE, data="test")

    def test_context_data_passed_through(self) -> None:
        received: dict = {}
        register_hook(HookEvent.ON_QUERY, lambda **kw: received.update(kw))
        fire_hook(HookEvent.ON_QUERY, query="hello", results=["r1"])
        assert received == {"query": "hello", "results": ["r1"]}

    def test_exception_in_callback_does_not_propagate(self) -> None:
        def bad_hook(**kw: object) -> None:
            raise RuntimeError("boom")

        register_hook(HookEvent.POST_COMPILE, bad_hook)
        good_called: list[bool] = []
        register_hook(HookEvent.POST_COMPILE, lambda **kw: good_called.append(True))
        fire_hook(HookEvent.POST_COMPILE)
        assert good_called == [True]

    def test_clear_hooks(self) -> None:
        calls: list[str] = []
        register_hook(HookEvent.ON_LINT, lambda **kw: calls.append("fired"))
        clear_hooks()
        fire_hook(HookEvent.ON_LINT)
        assert calls == []

    def test_event_isolation(self) -> None:
        compile_calls: list[str] = []
        capture_calls: list[str] = []
        register_hook(HookEvent.POST_COMPILE, lambda **kw: compile_calls.append("c"))
        register_hook(HookEvent.ON_CAPTURE, lambda **kw: capture_calls.append("cap"))
        fire_hook(HookEvent.POST_COMPILE)
        assert compile_calls == ["c"]
        assert capture_calls == []


class TestLoadFromConfig:
    def setup_method(self) -> None:
        clear_hooks()

    def teardown_method(self) -> None:
        clear_hooks()

    def test_empty_config_loads_nothing(self) -> None:
        count = load_hooks_from_config({})
        assert count == 0

    def test_missing_hooks_key(self) -> None:
        count = load_hooks_from_config({"compile": {}})
        assert count == 0

    def test_unknown_event_skipped(self) -> None:
        count = load_hooks_from_config({"hooks": {"unknown_event": ["os:exit"]}})
        assert count == 0

    def test_invalid_path_skipped(self) -> None:
        count = load_hooks_from_config({"hooks": {"post_compile": ["no_colon_here"]}})
        assert count == 0

    def test_import_failure_skipped(self) -> None:
        count = load_hooks_from_config(
            {"hooks": {"post_compile": ["nonexistent_module:callback"]}},
        )
        assert count == 0

    def test_valid_hook_loaded(self) -> None:
        config = {"hooks": {"on_query": ["tests.helpers.hooks:query_logger"]}}
        count = load_hooks_from_config(config)
        assert count == 1

        fired: list[bool] = []

        def my_query_hook(**kw: object) -> None:
            fired.append(True)

        register_hook(HookEvent.ON_QUERY, my_query_hook)
        fire_hook(HookEvent.ON_QUERY, query="test")
        assert fired == [True]
