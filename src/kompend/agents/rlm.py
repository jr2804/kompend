"""RLM (Recursive Language Model) capability — code exploration for KOMPEND agents.

Wraps the REPL toolset from ``pydantic-ai-rlm`` (vstorm-co) as a pydantic-ai
:class:`AbstractCapability` so KOMPEND agents can explore large deps contexts
programmatically instead of packing everything into the prompt.

The package's own ``create_rlm_toolset`` is typed for ``RLMDependencies``;
KOMPEND agents use their own deps dataclasses (``QueryDeps``, ``ToolGenerationDeps``,
…). This module owns the thin tool glue: it reads the context from a
configurable deps attribute and reuses the package's
:class:`~pydantic_ai_rlm.repl.REPLEnvironment` for the sandbox, context
loading, and the ``llm_query`` sub-model path.

Upstream gaps we work around (both open in vstorm-co/pydantic-ai-rlm):

- Recursion is not implemented (issue #4) — ``llm_query`` is a plain
  sub-model call. Phase 2 adds ``rlm_query`` depth-limited recursion here.
- ``llm_query`` breaks when the REPL ``execute`` runs on the main event-loop
  thread ("This event loop is already running"). The toolset runs ``execute``
  in a thread executor, so the package's ``model_request_sync`` works; keep
  that thread offload (see :meth:`RLMCapability.get_toolset`).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast, override

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_rlm.dependencies import RLMConfig
from pydantic_ai_rlm.repl import REPLEnvironment
from pydantic_ai_rlm.utils import format_repl_result

from kompend.agents.types import PipelineAgentDeps
from kompend.logging import get_logger

#: Chars sampled from each end of the context when fingerprinting a recursive
#: call ("have I seen this call already?"). A stable prefix+suffix pair is
#: enough to identify the call, without hashing a whole context per call.
_FINGERPRINT_SAMPLE_CHARS = 100

_EXECUTE_CODE_DESCRIPTION = """\
Execute Python code in a sandboxed REPL environment.

## Environment
- A `context` variable is pre-loaded with the data to analyze
- Variables persist between executions within the same session
- Use print() to display output

## When to Use
- Analyzing or processing structured data (JSON, dicts, lists)
- Performing calculations or data transformations
- Extracting specific information from large datasets

## Best Practices
1. Start by exploring the context: `print(type(context))`, `print(len(context))`
2. Break complex operations into smaller steps
3. Use print() liberally to understand intermediate results
4. Handle errors gracefully with try/except

## Available Functions
- `llm_query(prompt)`: Query a sub-LLM for reasoning assistance (if configured)
- `rlm_query(prompt, context_slice)`: Recursively analyze a slice with a
  child agent that has its own REPL (if configured)
- Important: Do not use `llm_query` in the first code execution. Use it only
  after you have explored the context and identified specific sections that
  need semantic analysis.

## Example
```python
print(f"Context type: {type(context)}")
print(f"Length: {len(context)}")
# ... slice and analyze the relevant section ...
```
"""

#: Default total recursive ``rlm_query`` calls allowed per root agent run.
_DEFAULT_NODE_BUDGET: int = 16

logger = get_logger(__name__)

#: Per-run REPL environments keyed by ``id(ctx.deps)`` — one REPL per agent
#: run, mirroring the package's module-global registry. Guards (recursion
#: budgets) live alongside, one per ROOT run, shared across the child tree.
_repl_registry: dict[int, REPLEnvironment] = {}
_guard_registry: dict[int, _RLMRunGuard] = {}
_repl_lock = threading.Lock()


@dataclass
class _RLMRunGuard:
    """Guardrails shared across one recursive RLM call tree.

    Tracks the total node budget and a cycle-detection set over
    ``(depth, prompt, context-fingerprint)`` keys — identical prompt+context
    at the same depth is a loop; the same prompt over *different* slices
    (the common map pattern) stays allowed. Depth itself is tracked per
    capability (not here) since siblings at one level share the guard.
    """

    node_budget: int = _DEFAULT_NODE_BUDGET
    nodes_used: int = 0
    seen: set[int] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)


class RLMCapability(AbstractCapability[Any]):
    """Adds RLM code exploration to a KOMPEND agent.

    Supplies the ``execute_code`` tool (REPL sandbox with the deps context
    pre-loaded) and the RLM mode instructions. The sub-model, when set,
    powers ``llm_query()`` inside the REPL; wrap it in
    :class:`~kompend.agents.kompend_model.KompendModel` for usage tracking.
    """

    def __init__(
        self,
        *,
        context_field: str,
        code_timeout: float = 60.0,
        sub_model: Model | None = None,
        max_tool_calls: int = 50,
        main_model: Model | None = None,
        depth: int = 0,
        node_budget: int = _DEFAULT_NODE_BUDGET,
        guard: _RLMRunGuard | None = None,
        source_id: str = "",
    ) -> None:
        """Initialize the capability.

        Args:
            context_field: Name of the deps attribute holding the document
                context (e.g. ``"context"`` for QueryDeps, ``"content"`` for
                ToolGenerationDeps, ``"text"`` for PipelineAgentDeps).
            code_timeout: Per-execution timeout in seconds.
            sub_model: Model for ``llm_query()`` inside the REPL. ``None``
                disables ``llm_query`` (structural exploration only).
            max_tool_calls: ``UsageLimits.tool_calls_limit`` for the run.
            main_model: The agent's top-level model; powers child agents in
                recursive ``rlm_query`` calls. ``None`` disables recursion.
            depth: Remaining recursion depth for ``rlm_query`` (0 = off).
            node_budget: Total ``rlm_query`` calls allowed per root run.
            guard: Shared guardrails — passed down by parent capabilities;
                created fresh for root runs.
            source_id: Provenance label for child runs in logs/debug dumps.
        """
        self._context_field = context_field
        self._code_timeout = code_timeout
        self._sub_model = sub_model
        self._max_tool_calls = max_tool_calls
        self._main_model = main_model
        self._depth = depth
        self._node_budget = node_budget
        self._guard = guard
        self._source_id = source_id
        self._executions = 0
        self._queries = 0

    @property
    def max_tool_calls(self) -> int:
        """Tool-call limit for the agent run (``UsageLimits``)."""
        return self._max_tool_calls

    def get_toolset(self) -> FunctionToolset[Any] | None:
        """Return the RLM ``execute_code`` toolset, or None."""
        toolset: FunctionToolset[Any] = FunctionToolset(id="kompend-rlm")
        cap = self

        @toolset.tool(description=_EXECUTE_CODE_DESCRIPTION)
        async def execute_code(ctx: RunContext[Any], code: str) -> str:
            context = getattr(ctx.deps, cap._context_field)
            repl = _get_repl(
                id(ctx.deps),
                context,
                cap._sub_model,
                cap._code_timeout,
                main_model=cap._main_model,
                depth=cap._depth,
                node_budget=cap._node_budget,
                guard=cap._guard,
                source_id=getattr(ctx.deps, "source_id", "") or cap._source_id,
            )
            cap._executions += 1
            # REPL.execute is sync and llm_query needs a thread without a
            # running event loop — offload to the executor (upstream bug #4
            # workaround, see module docstring).
            loop = asyncio.get_running_loop()
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, repl.execute, code),
                    timeout=cap._code_timeout,
                )
                return format_repl_result(result)
            except TimeoutError:
                return f"Error: Code execution timed out after {cap._code_timeout} seconds."
            except Exception as exc:  # noqa: BLE001 — surfaced to the model, not fatal
                return f"Error executing code: {exc!s}"

        return toolset

    def get_instructions(self) -> str | None:
        """Return the RLM mode instruction block."""
        return build_rlm_instructions(
            sub_model=self._sub_model is not None,
            recursive=self._depth > 0 and self._main_model is not None,
        )

    @override
    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Log RLM activity; observe-only (no result mutation)."""
        if tool_def.name == "execute_code":
            logger.debug("RLM execute_code returned %d chars", len(str(result)))
        return result


def _get_repl(
    deps_id: int,
    context: Any,
    sub_model: Model | None,
    code_timeout: float,
    *,
    main_model: Model | None = None,
    depth: int = 0,
    node_budget: int = _DEFAULT_NODE_BUDGET,
    guard: _RLMRunGuard | None = None,
    source_id: str = "",
) -> REPLEnvironment:
    """Get or create the REPL environment for one agent run.

    When ``depth > 0`` and a main model is available, the recursive
    ``rlm_query`` builtin is injected into the REPL (upstream issue #4).
    Child capabilities pass the root ``guard`` down so the whole call tree
    shares one budget and cycle-detection set.
    """
    with _repl_lock:
        repl = _repl_registry.get(deps_id)
        if repl is None:
            # The dependency annotates ``sub_model`` as ``str | None`` but its
            # REPL forwards the value to ``model_request_sync``, which accepts
            # ``Model | str`` — a KompendModel flows through unchanged.
            repl = REPLEnvironment(
                context=context,
                config=RLMConfig(
                    sub_model=cast("str | None", sub_model),
                    code_timeout=code_timeout,
                ),
            )
            _repl_registry[deps_id] = repl
            if guard is None:
                guard = _RLMRunGuard(node_budget=node_budget)
                _guard_registry[deps_id] = guard
            if depth > 0 and main_model is not None:
                repl.locals["rlm_query"] = _make_rlm_query(
                    guard=guard,
                    my_depth=depth,
                    main_model=main_model,
                    sub_model=sub_model,
                    code_timeout=code_timeout,
                    parent_context=context,
                    source_id=source_id,
                )
    return repl


def _make_rlm_query(
    *,
    guard: _RLMRunGuard,
    my_depth: int,
    main_model: Model,
    sub_model: Model | None,
    code_timeout: float,
    parent_context: Any,
    source_id: str,
) -> Callable[[str, Any], str]:
    """Build the recursive ``rlm_query(prompt, context=None)`` REPL builtin.

    Spawns a child agent (same main model, its own REPL over the given
    slice) that can itself use ``llm_query`` — and ``rlm_query`` again at
    ``my_depth - 1`` — until the depth budget is exhausted. At depth 0 no
    rlm_query is injected, so children fall back to plain exploration +
    ``llm_query``. Runs inside the parent REPL's worker thread, so
    ``run_sync`` is safe there.
    """

    def rlm_query(prompt: str, context: Any = None) -> str:
        """Recursively analyze a context slice with a child RLM agent.

        Args:
            prompt: The question for the child agent.
            context: The slice to analyze. Defaults to the full parent
                context when omitted.

        Returns:
            The child agent's final answer text.
        """
        ctx = str(context) if context is not None else str(parent_context)
        fingerprint = hash(
            (my_depth, prompt, len(ctx), ctx[:_FINGERPRINT_SAMPLE_CHARS], ctx[-_FINGERPRINT_SAMPLE_CHARS:] if len(ctx) > 2 * _FINGERPRINT_SAMPLE_CHARS else "")
        )
        with guard.lock:
            if guard.nodes_used >= guard.node_budget:
                return f"Error: RLM node budget exhausted ({guard.node_budget} recursive calls) — synthesize an answer from the results gathered so far."
            if fingerprint in guard.seen:
                return "Error: identical recursive query already issued (cycle) — vary the prompt or context slice."
            guard.seen.add(fingerprint)
            guard.nodes_used += 1

        child = RLMCapability(
            context_field="text",
            code_timeout=code_timeout,
            sub_model=sub_model,
            main_model=main_model,
            depth=my_depth - 1,
            guard=guard,
            source_id=f"{source_id}#rlm{guard.nodes_used}",
        )
        child_agent = Agent(main_model, output_type=str, capabilities=[child], name="rlm-child-agent")
        deps = PipelineAgentDeps(text=ctx, source_id=child._source_id)
        result = child_agent.run_sync(prompt, deps=deps)
        return str(result.output)

    return rlm_query


def cleanup_rlm_repls() -> None:
    """Drop all cached REPL environments (tests, long-running processes)."""
    with _repl_lock:
        _repl_registry.clear()


def build_rlm_instructions(*, sub_model: bool, recursive: bool = False) -> str:
    """RLM mode instructions appended to the agent system prompt.

    Directive, not optional: when RLM is enabled the document context is
    replaced by a placeholder in the prompt, so the model cannot answer
    without calling ``execute_code`` first.
    """
    steps = [
        "The document context is NOT inlined in this prompt — it is pre-loaded as the `context` variable of the `execute_code` REPL.",
        "You MUST call `execute_code` to explore `context` before answering. Start with: print(type(context)); print(len(context))",
    ]
    if sub_model:
        steps.append(
            "Use `llm_query(prompt)` to delegate semantic analysis of a small slice to the sub-model (it is cheaper than adding context).",
        )
    if recursive:
        steps.append(
            "Use `rlm_query(prompt, context_slice)` to recursively analyze a "
            "larger slice with a child agent that has its own REPL — reserve it "
            "for slices that need code exploration, not simple lookups.",
        )
    steps.append("Keep code concise and exploratory; only return small, relevant slices.")
    steps.append("Synthesize your final answer normally (as the agent's output).")
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    return "## RLM Mode\n\n" + numbered + "\n"
