"""ACP agent CLI — run kompend as an ACP-compatible agent.

Usage::

    kompend agent [--mcp-port <port>]

Starts an ACP agent on stdio via ``pydantic-ai-harness``'s
``run_acp_stdio_sync``. The harness handles the ACP protocol, session
management, streaming, tool approval, and cancellation. kompend provides
the agent, tools, and per-session configuration.

The agent stack is imported at module level: Cyclopts resolves this module
on demand, so nothing is paid at ``kompend --help`` time.

The MCP companion lifecycle (start the server task, run the ACP session,
cancel and drain the task) lives behind :func:`_run_with_mcp_sidecar`, whose
three collaborators are injectable. The command signature stays the CLI's.

The companion cannot share the main thread's loop: ``run_acp_stdio_sync`` calls
``asyncio.run(...)`` and therefore owns — and finally closes — that loop, while
``asyncio.create_task`` needs a running one. So the server gets a private loop
on a daemon thread and the ACP session keeps the main thread; the two loops
never contend. The thread is a daemon so that a server which ignores its
cancellation cannot hold the interpreter open.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Callable, Coroutine
from typing import Any, Final, Protocol

from pydantic_ai import Agent
from pydantic_ai_harness.experimental.acp import run_acp_stdio_sync

from kompend.acp import build_kompend_agent, kompend_session_config
from kompend.cli.args.serving import AgentMcpPort
from kompend.logging import get_logger
from kompend.mcp.server import create_mcp_server

#: Longest the teardown waits for the cancelled MCP server task, and for its
#: loop thread to finish.
_TEARDOWN_TIMEOUT_S: Final[float] = 5.0

#: Name of the companion server's loop thread, for tracebacks and thread dumps.
_COMPANION_THREAD_NAME: Final[str] = "kompend-mcp-companion"

logger = get_logger(__name__)


class _McpServer(Protocol):
    """The slice of the MCP server this module drives."""

    def run_http_async(self, *, port: int, transport: str) -> Coroutine[Any, Any, None]:
        """Serve the MCP endpoint until the coroutine is cancelled."""


class _McpTaskRunner(Protocol):
    """Schedules the MCP server coroutine and drains it at teardown.

    The handle :meth:`start` returns is opaque to the caller: the production
    runner hands back its own record, a test double may hand back a sentinel
    only it understands. :meth:`stop` cancels the work and waits for it,
    raising whatever the cancellation surfaces — including the standard
    library's cancellation and timeout outcomes, which share no base class with
    the asyncio ones.
    """

    def start(self, server: Coroutine[Any, Any, None]) -> object:
        """Run *server* as a background task and return its handle."""

    def stop(self, handle: object, *, timeout: float) -> None:
        """Cancel *handle* and wait up to *timeout* seconds for it to finish."""


class _CompanionLoop:
    """One MCP server coroutine, its private event loop, and that loop's thread."""

    __slots__ = ("future", "loop", "thread")

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        thread: threading.Thread,
        future: concurrent.futures.Future[None],
    ) -> None:
        self.loop = loop
        self.thread = thread
        self.future = future


class _ThreadLoopTaskRunner:
    """The production runner: the server gets a loop and a thread of its own.

    Sharing the main thread's loop is not possible: the ACP session calls
    ``asyncio.run``, which owns that loop for the whole session and closes it on
    the way out. A private loop keeps the companion serving for as long as the
    session lasts, and the daemon thread guarantees a wedged server cannot keep
    the process alive past the bounded teardown wait.
    """

    @staticmethod
    def start(server: Coroutine[Any, Any, None]) -> object:
        """Start a private loop on a daemon thread and schedule *server* on it."""
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, name=_COMPANION_THREAD_NAME, daemon=True)
        thread.start()
        future = asyncio.run_coroutine_threadsafe(server, loop)
        return _CompanionLoop(loop=loop, thread=thread, future=future)

    @staticmethod
    def stop(handle: object, *, timeout: float) -> None:
        """Cancel the server, drain the loop, then stop it and join the thread."""
        companion = _as_companion(handle)
        try:
            _cancel_and_drain(companion.future, timeout=timeout)
        finally:
            _drain_remaining(companion.loop, timeout=timeout)
            companion.loop.call_soon_threadsafe(companion.loop.stop)
            companion.thread.join(timeout=timeout)
            if not companion.thread.is_alive():
                companion.loop.close()


def _cancel_and_drain(future: concurrent.futures.Future[None], *, timeout: float) -> None:
    """Cancel the server unless it already finished, then wait for it.

    Order matters: cancelling a future whose coroutine has not started yet
    replaces its outcome with a cancellation, so a companion that died on
    startup (a taken port, say) would be reported as a quiet normal exit. Ask
    for the result first — a finished future answers immediately, with its own
    error — and only cancel one that is still running.
    """
    try:
        future.result(timeout=0)
    except concurrent.futures.TimeoutError:
        future.cancel()
        future.result(timeout=timeout)


async def _drain_tasks() -> None:
    """Cancel every task on the companion loop except this one, and await them.

    A task that is already unwinding a cancellation is awaited, not cancelled
    again: a second cancel lands inside the task's ``finally`` block and the
    cleanup it was doing — closing a listener, for one — never finishes.
    """
    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    for task in tasks:
        if not task.cancelling():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _drain_remaining(loop: asyncio.AbstractEventLoop, *, timeout: float) -> None:
    """Cancel what the server left behind, while its loop is still running.

    A cancelled HTTP server leaves its app's lifespan and connection tasks
    pending. They are cancelled and awaited on the companion loop itself, which
    is the only thread allowed to touch them; stopping the loop first and
    draining afterwards would leave tasks that never run again, and CPython
    reports those as destroyed-but-pending. The wait is bounded, and a drain
    that overruns is not an error: the thread is a daemon.
    """
    try:
        asyncio.run_coroutine_threadsafe(_drain_tasks(), loop).result(timeout=timeout)
    except (TimeoutError, concurrent.futures.CancelledError):
        pass  # best effort: the loop thread is a daemon and cannot hold exit
    except RuntimeError as exc:
        logger.debug("companion loop was already closed during teardown: %s", exc)


def _as_companion(handle: object) -> _CompanionLoop:
    """Narrow a runner handle back to the record the thread runner created."""
    if not isinstance(handle, _CompanionLoop):
        msg = f"the thread-loop runner only stops its own handles, got {type(handle).__name__}"
        raise TypeError(msg)
    return handle


def agent_cmd(
    *,
    mcp_port: AgentMcpPort = None,
) -> None:
    """Run kompend as an ACP agent on stdio.

    Serves a PydanticAI agent over the Agent Client Protocol using
    ``pydantic-ai-harness``. The agent exposes kompend tools (compile,
    query, capture, config) and resolves per-session configuration
    from the workspace the editor opened.
    """
    agent = build_kompend_agent()

    if mcp_port is not None:
        _run_with_mcp_sidecar(agent, mcp_port)
    else:
        _run_sync(agent)


def _run_with_mcp_sidecar(
    agent: Agent[None, str],
    mcp_port: int,
    *,
    runner: _McpTaskRunner | None = None,
    mcp_server: _McpServer | None = None,
    run_agent: Callable[[], None] | None = None,
) -> None:
    """Run the ACP session with the MCP companion server alive alongside it.

    The server task is cancelled and drained in a ``finally``, so the teardown
    never masks the agent run's outcome. A cancelled task and an expired wait
    are the expected teardown outcomes and stay silent; any other failure is
    recorded, because a server that died on startup would otherwise leave the
    operator wondering why ``--mcp-port`` connects to nothing.

    Args:
        agent: The agent the ACP session serves.
        mcp_port: Port for the MCP-over-HTTP companion server.
        runner: Task lifecycle; defaults to the daemon-thread loop runner.
        mcp_server: Pre-built server; defaults to :func:`create_mcp_server`.
        run_agent: The blocking ACP run; defaults to :func:`_run_sync`.

    Raises:
        Exception: whatever the agent run raised — see above.
    """
    task_runner = runner if runner is not None else _ThreadLoopTaskRunner()
    server = mcp_server if mcp_server is not None else create_mcp_server()
    run = run_agent if run_agent is not None else lambda: _run_sync(agent)
    handle = task_runner.start(server.run_http_async(port=mcp_port, transport="streamable-http"))
    try:
        run()
    finally:
        try:
            task_runner.stop(handle, timeout=_TEARDOWN_TIMEOUT_S)
        except (asyncio.CancelledError, concurrent.futures.CancelledError, TimeoutError):
            pass  # expected teardown outcomes: the task is cancelled and the wait is bounded
        except Exception as exc:  # teardown must not mask the agent run, but failures are recorded
            logger.debug("mcp server task failed during shutdown: %s", exc)


def _run_sync(agent: Agent[None, str]) -> None:
    """Run the ACP agent synchronously."""
    run_acp_stdio_sync(
        agent,
        session_config=kompend_session_config,
        name="kompend",
        version="0.1.0",
    )
