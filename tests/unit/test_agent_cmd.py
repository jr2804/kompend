"""Tests for the ACP agent command's MCP companion lifecycle.

``kompend agent --mcp-port <port>`` runs two things at once: the ACP session on
stdio and an MCP-over-HTTP server in the background. The teardown that cancels
and drains the server task must never mask the agent run, and it must not
swallow a server that died on startup either — the operator would otherwise see
``--mcp-port`` accept a connection to nothing.

Most tests are driven through the seam
:func:`kompend.cli.cmds.agent._run_with_mcp_sidecar` injects, so they never
touch a real event loop, a socket, or the ACP protocol. :class:`TestThreadLoopTaskRunner`
closes the loop the other way: it starts the production runner for real and
proves that the companion serves on its own thread while the caller's thread
stays free — the property that makes ``--mcp-port`` work at all.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import socket
import threading
from collections.abc import Coroutine
from typing import Any, cast

import pytest
from pydantic_ai import Agent

from kompend.cli.cmds.agent import (
    _COMPANION_THREAD_NAME,
    _TEARDOWN_TIMEOUT_S,
    _CompanionLoop,
    _run_with_mcp_sidecar,
    _ThreadLoopTaskRunner,
    agent_cmd,
)

_AGENT_LOGGER = "kompend.cli.cmds.agent"
_SHUTDOWN_FAILURE_MARKER = "mcp server task failed during shutdown"

#: Generous bound for thread/socket handshakes; every wait below is on this, not
#: on a fixed sleep, so a slow machine cannot turn a pass into a flake.
_WAIT_S = 5.0


class _FakeServer:
    """Records the HTTP-server call the command makes, without serving."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def run_http_async(self, *, port: int, transport: str) -> Coroutine[Any, Any, None]:
        """Record the port/transport pair and hand back an unstarted coroutine."""
        self.calls.append((port, transport))
        return _never()


class _FakeRunner:
    """Records the task lifecycle and replays a chosen teardown outcome."""

    def __init__(self, *, stop_error: BaseException | None = None) -> None:
        self.stop_error = stop_error
        self.events: list[str] = []
        self.timeouts: list[float] = []

    def start(self, server: Coroutine[Any, Any, None]) -> object:
        """Release the coroutine and return an opaque handle."""
        server.close()
        self.events.append("start")
        return "handle"

    def stop(self, handle: object, *, timeout: float) -> None:
        """Record the wait bound, then replay the configured outcome."""
        self.events.append("stop")
        self.timeouts.append(timeout)
        assert handle == "handle", f"stop received an unexpected handle: {handle!r}"
        if self.stop_error is not None:
            raise self.stop_error


class TestSidecarTeardown:
    def test_a_cancelled_mcp_task_stays_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        """Cancellation is the expected teardown outcome: the command ends quietly."""
        runner = _FakeRunner(stop_error=asyncio.CancelledError())

        with caplog.at_level(logging.DEBUG, logger=_AGENT_LOGGER):
            _run_with_mcp_sidecar(
                _agent(),
                8000,
                runner=runner,
                mcp_server=_FakeServer(),
                run_agent=lambda: None,
            )

        assert runner.events == ["start", "stop"], "the teardown must run after the agent run"
        assert _shutdown_messages(caplog) == []

    def test_the_companion_loop_cancellation_stays_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        """The thread runner cancels a ``concurrent.futures`` future, not an asyncio task.

        That cancellation type is a plain ``Exception`` subclass with no relation
        to :class:`asyncio.CancelledError`, so it needs its own clause in the
        teardown: without it, every normal exit would log a shutdown failure.
        """
        runner = _FakeRunner(stop_error=concurrent.futures.CancelledError())

        with caplog.at_level(logging.DEBUG, logger=_AGENT_LOGGER):
            _run_with_mcp_sidecar(
                _agent(),
                8000,
                runner=runner,
                mcp_server=_FakeServer(),
                run_agent=lambda: None,
            )

        assert _shutdown_messages(caplog) == []

    def test_a_timed_out_mcp_task_stays_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        """The bounded wait expiring is expected too — no server, no noise."""
        runner = _FakeRunner(stop_error=TimeoutError())

        with caplog.at_level(logging.DEBUG, logger=_AGENT_LOGGER):
            _run_with_mcp_sidecar(
                _agent(),
                8000,
                runner=runner,
                mcp_server=_FakeServer(),
                run_agent=lambda: None,
            )

        assert _shutdown_messages(caplog) == []

    def test_a_failing_mcp_task_records_the_reason(self, caplog: pytest.LogCaptureFixture) -> None:
        """An unexpected server failure is recorded, never swallowed."""
        runner = _FakeRunner(stop_error=RuntimeError("port 8000 already in use"))

        with caplog.at_level(logging.DEBUG, logger=_AGENT_LOGGER):
            _run_with_mcp_sidecar(
                _agent(),
                8000,
                runner=runner,
                mcp_server=_FakeServer(),
                run_agent=lambda: None,
            )

        messages = _shutdown_messages(caplog)
        assert len(messages) == 1, f"expected exactly one shutdown failure, got {messages}"
        assert "port 8000 already in use" in messages[0]

    def test_a_teardown_failure_does_not_mask_the_agent_run(self, caplog: pytest.LogCaptureFixture) -> None:
        """The agent run's exception wins; the teardown failure is only recorded."""
        runner = _FakeRunner(stop_error=RuntimeError("teardown exploded"))

        def _fail() -> None:
            raise RuntimeError("acp session failed")

        with caplog.at_level(logging.DEBUG, logger=_AGENT_LOGGER), pytest.raises(RuntimeError, match="acp session failed"):
            _run_with_mcp_sidecar(
                _agent(),
                8000,
                runner=runner,
                mcp_server=_FakeServer(),
                run_agent=_fail,
            )

        messages = _shutdown_messages(caplog)
        assert len(messages) == 1, f"expected one recorded teardown failure, got {messages}"
        assert "teardown exploded" in messages[0]


class TestSidecarOrdering:
    def test_the_server_starts_before_the_agent_runs(self) -> None:
        """The companion server is scheduled first and drained last."""
        runner = _FakeRunner()
        order: list[str] = []
        server = _FakeServer()

        def _run() -> None:
            order.append("run")

        _run_with_mcp_sidecar(_agent(), 8123, runner=runner, mcp_server=server, run_agent=_run)

        assert runner.events == ["start", "stop"]
        assert order == ["run"]
        assert server.calls == [(8123, "streamable-http")]

    def test_the_teardown_wait_is_bounded(self) -> None:
        """The drain waits at most the module's timeout, so a wedged server cannot hang exit."""
        runner = _FakeRunner()

        _run_with_mcp_sidecar(_agent(), 8000, runner=runner, mcp_server=_FakeServer(), run_agent=lambda: None)

        assert runner.timeouts == [_TEARDOWN_TIMEOUT_S]

    def test_the_default_collaborators_are_the_real_ones(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no injection the seam calls the module's own server and ACP run."""
        calls: list[str] = []

        class _Server:
            def run_http_async(self, *, port: int, transport: str) -> Coroutine[Any, Any, None]:
                calls.append(f"serve {port} {transport}")
                return _never()

        def _run_sync(agent: Agent[None, str]) -> None:
            calls.append("run_sync")

        monkeypatch.setattr("kompend.cli.cmds.agent.create_mcp_server", _Server)
        monkeypatch.setattr("kompend.cli.cmds.agent._run_sync", _run_sync)

        _run_with_mcp_sidecar(_agent(), 9000, runner=_FakeRunner())

        assert calls == ["serve 9000 streamable-http", "run_sync"]


class TestAgentCommand:
    def test_without_a_port_only_the_agent_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No ``--mcp-port`` means no server: the command is one ACP run."""
        calls: list[str] = []
        monkeypatch.setattr("kompend.cli.cmds.agent.build_kompend_agent", _agent)
        monkeypatch.setattr("kompend.cli.cmds.agent._run_sync", lambda _agent: calls.append("run_sync"))
        monkeypatch.setattr("kompend.cli.cmds.agent.create_mcp_server", _unexpected_call)

        agent_cmd()

        assert calls == ["run_sync"], f"expected only the ACP run, got {calls}"

    def test_with_a_port_the_sidecar_owns_the_lifecycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--mcp-port`` hands the run to the companion-server seam."""
        calls: list[tuple[str, int]] = []
        agent = _agent()
        monkeypatch.setattr("kompend.cli.cmds.agent.build_kompend_agent", lambda: agent)
        monkeypatch.setattr(
            "kompend.cli.cmds.agent._run_with_mcp_sidecar",
            lambda built, port: calls.append((f"sidecar for {built is agent}", port)),
        )

        agent_cmd(mcp_port=8000)

        assert calls == [("sidecar for True", 8000)]


class TestThreadLoopTaskRunner:
    """The production runner, started for real.

    This is the regression test for the defect the ticket was filed about: the
    companion used to be scheduled on the main thread's loop with
    ``asyncio.create_task``, which raises ``RuntimeError: no running event loop``
    from a synchronous command — and would have been starved anyway, because the
    ACP session runs its own ``asyncio.run`` loop. The runner must serve on a
    thread of its own.
    """

    def test_the_default_runner_serves_on_its_own_thread(self) -> None:
        """A real server accepts a connection while the caller's thread stays free."""
        runner = _ThreadLoopTaskRunner()
        ready = threading.Event()
        drained = threading.Event()
        seen: dict[str, object] = {}

        async def _serve() -> None:
            seen["thread"] = threading.current_thread().name

            async def _handle(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                writer.close()

            server = await asyncio.start_server(_handle, host="127.0.0.1", port=0)
            seen["port"] = server.sockets[0].getsockname()[1]
            ready.set()
            try:
                await asyncio.sleep(3600)
            finally:
                server.close()
                await server.wait_closed()
                drained.set()

        handle = cast(_CompanionLoop, runner.start(_serve()))
        port = 0
        try:
            assert ready.wait(_WAIT_S), "the companion never started serving"
            assert seen["thread"] == _COMPANION_THREAD_NAME
            assert seen["thread"] != threading.current_thread().name
            port = cast(int, seen["port"])
            with socket.create_connection(("127.0.0.1", port), timeout=_WAIT_S):
                pass  # the companion accepted while this thread was not running any loop
            assert not drained.is_set(), "the server must still be serving before the teardown"
        finally:
            # A cancelled live server surfaces the standard library's cancellation;
            # the seam is what treats that as the expected outcome.
            with pytest.raises(concurrent.futures.CancelledError):
                runner.stop(handle, timeout=_WAIT_S)

        assert drained.is_set(), "the teardown must cancel the server and let it unwind"
        assert not handle.thread.is_alive(), "the companion loop thread must end"
        # A bind probe, not a connect: on Windows a connect to a closed port may
        # time out instead of being refused, which would fail either way.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", port))
        finally:
            probe.close()

    def test_a_server_that_dies_on_startup_raises_the_real_reason(self) -> None:
        """A startup failure is reported at teardown, not replaced by a cancellation."""

        async def _fail() -> None:
            raise RuntimeError("port 8000 already in use")

        runner = _ThreadLoopTaskRunner()
        handle = cast(_CompanionLoop, runner.start(_fail()))
        _wait_done(handle)

        with pytest.raises(RuntimeError, match="port 8000 already in use"):
            runner.stop(handle, timeout=_WAIT_S)

    def test_it_cancels_a_server_that_is_still_serving(self) -> None:
        """A live companion is cancelled and drained, then its loop thread ends."""

        async def _serve() -> None:
            await asyncio.sleep(3600)

        runner = _ThreadLoopTaskRunner()
        handle = cast(_CompanionLoop, runner.start(_serve()))

        with pytest.raises(concurrent.futures.CancelledError):
            runner.stop(handle, timeout=_WAIT_S)

        assert handle.future.done()
        assert not handle.thread.is_alive()

    def test_it_rejects_a_handle_it_did_not_create(self) -> None:
        """A foreign handle is a programming error, not a silent no-op."""
        with pytest.raises(TypeError, match="only stops its own handles"):
            _ThreadLoopTaskRunner().stop("handle", timeout=1.0)


async def _never() -> None:
    """Stand-in for the server coroutine; the fakes close it instead of running it."""
    raise AssertionError("the command must not touch the MCP server in this test")


def _unexpected_call(*args: object, **kwargs: object) -> None:
    """Fail loudly if the sidecar is built without a port."""
    raise AssertionError(f"the MCP companion must not be built here: {args} {kwargs}")


def _agent() -> Agent[None, str]:
    """An agent placeholder: the seam never reads it, the CLI builds the real one."""
    return cast(Agent[None, str], object())


def _wait_done(handle: _CompanionLoop) -> None:
    """Block until the companion's coroutine has finished, or fail the test."""
    deadline = threading.Event()
    waited = 0.0
    while not handle.future.done():
        assert waited < _WAIT_S, "the companion coroutine never finished"
        deadline.wait(0.01)
        waited += 0.01


def _shutdown_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Every recorded message that reports a teardown failure."""
    return [record.getMessage() for record in caplog.records if _SHUTDOWN_FAILURE_MARKER in record.getMessage()]
