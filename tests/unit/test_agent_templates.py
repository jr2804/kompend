"""Systemic guard for agent prompt templates and their tool wiring.

Each :class:`KnoxAgentBase` subclass declares its prompt paths through the
``template_prefix``, ``template_name`` and ``system_template_name`` ClassVars.
The base ``render_user_prompt`` / ``render_system_prompt`` build the path as
``template_prefix + <name>``; a prefix that does not match the template's
directory fails only when the agent first renders its prompt — at runtime, not
at import.

For every agent, this module checks:

1. each base-managed declared template path resolves through the real
   :class:`TemplateLoader` (catches a moved template or a wrong prefix);
2. each base-managed template body is valid Jinja2;
3. the system body renders; and for agentic agents, the ``name(`` tool calls
   it declares equal the tools the agent actually registers (catches a prompt
   that tells the model to call tools the agent does not have — the
   ``d443c583`` dropped-factory regression).

Agents that override ``render_user_prompt`` / ``render_system_prompt`` own their
path (they may supply deps variables or hardcode a directory), so their declared
ClassVars are not checked. Tool-declaration tokens are only asserted for
``Agentic*`` agents, because a tool-less system prompt can contain prose that
looks like a call (e.g. "first author(s)"); an ``Agentic*`` agent whose prompt
declares tools but registers none still fails, which is the regression class.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from collections.abc import Iterator

import pytest
from jinja2 import Environment, UndefinedError

import knox.agents as agents_package
from knox.agents.base import KnoxAgentBase
from knox.core.templates import get_template_loader

_TOOL_CALL_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\(")
_MODEL = "ollama/gpt-oss:20b"
_BASE_RENDER_USER = KnoxAgentBase.render_user_prompt
_BASE_RENDER_SYSTEM = KnoxAgentBase.render_system_prompt

_loader = get_template_loader()


def _iter_agent_classes() -> Iterator[type[KnoxAgentBase]]:
    """Yield every KnoxAgentBase subclass defined inside ``knox.agents.*``."""
    seen: set[type[KnoxAgentBase]] = set()
    for module_info in pkgutil.iter_modules(agents_package.__path__):
        module = importlib.import_module(f"{agents_package.__name__}.{module_info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj is KnoxAgentBase or not issubclass(obj, KnoxAgentBase):
                continue
            if obj.__module__ != module.__name__:
                continue  # imported into this module, not defined here
            if obj not in seen:
                seen.add(obj)
                yield obj


_AGENTS: list[type[KnoxAgentBase]] = sorted(_iter_agent_classes(), key=lambda cls: cls.__name__)


@pytest.mark.parametrize("agent", _AGENTS, ids=lambda cls: cls.__name__)
def test_declared_templates_resolve(agent: type[KnoxAgentBase]) -> None:
    user_template, system_template = _declared_templates(agent)
    declared = [path for path in (user_template, system_template) if path]
    if not declared:
        pytest.skip(f"{agent.__name__} declares no base-managed template")
    for path in declared:
        _loader.locate(path)  # raises FileNotFoundError when the declared path is wrong


@pytest.mark.parametrize("agent", _AGENTS, ids=lambda cls: cls.__name__)
def test_declared_templates_parse(agent: type[KnoxAgentBase]) -> None:
    user_template, system_template = _declared_templates(agent)
    declared = [path for path in (user_template, system_template) if path]
    if not declared:
        pytest.skip(f"{agent.__name__} declares no base-managed template")
    for path in declared:
        # Valid Jinja2 (syntax + expression parse); rendering needs runtime deps.
        Environment().parse(_loader.get_body(path))  # noqa: S701 — AST parse only


@pytest.mark.parametrize("agent", _AGENTS, ids=lambda cls: cls.__name__)
def test_system_template_renders(agent: type[KnoxAgentBase]) -> None:
    _, system_template = _declared_templates(agent)
    if not system_template:
        pytest.skip(f"{agent.__name__} declares no base-managed system template")
    try:
        rendered = _loader.render(system_template)
    except (UndefinedError, ValueError) as exc:
        # The base render passes no variables; a template that needs runtime
        # variables is a separate concern from the path/tool checks here.
        pytest.skip(f"{agent.__name__} system template needs variables: {exc}")
    assert isinstance(rendered, str)


@pytest.mark.parametrize("agent", [cls for cls in _AGENTS if cls.__name__.startswith("Agentic")], ids=lambda cls: cls.__name__)
def test_agentic_system_tools_match_registered(agent: type[KnoxAgentBase]) -> None:
    """The tools the system prompt names must be exactly the tools registered."""
    _, system_template = _declared_templates(agent)
    if not system_template:
        pytest.skip(f"{agent.__name__} declares no base-managed system template")
    system_body = _loader.render(system_template)
    registered = _registered_tool_names(_build_agent_instance(agent))
    declared = _declared_tool_names(system_body)
    assert declared == registered, f"{agent.__name__} system prompt declares {declared}, agent registers {registered}"


def _declared_templates(agent: type[KnoxAgentBase]) -> tuple[str, str]:
    """Return base-managed ``(user_template, system_template)`` paths.

    Empty when the agent declares no template *or* overrides the render method
    that would use it — in both cases the base does not resolve this path.
    """
    prefix = agent.template_prefix
    user_template = prefix + agent.template_name if agent.template_name and agent.render_user_prompt is _BASE_RENDER_USER else ""
    system_template = prefix + agent.system_template_name if agent.system_template_name and agent.render_system_prompt is _BASE_RENDER_SYSTEM else ""
    return user_template, system_template


def _build_agent_instance(agent: type[KnoxAgentBase]) -> KnoxAgentBase:
    """Construct an agent instance, preferring its ``make_*`` factory.

    Tools are attached by a factory (``tools=build_*_tools()``), so a direct
    construction would hide the wiring. Scan the agent's module for a factory
    that accepts a ``model_string`` and returns this class; fall back to direct
    construction for agents with no factory (their tool set is empty).
    """
    module = importlib.import_module(agent.__module__)
    for name, candidate in vars(module).items():
        if not inspect.isfunction(candidate) or not name.startswith("make_"):
            continue
        parameters = inspect.signature(candidate).parameters
        if "model_string" not in parameters and not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            continue
        instance = candidate(model_string=_MODEL)
        if type(instance) is agent:
            return instance
    return agent(model_string=_MODEL)


def _registered_tool_names(agent: KnoxAgentBase) -> list[str]:
    """Tool names the built pydantic-ai agent actually registers."""
    return sorted(agent._build_agent()._function_toolset.tools.keys())  # noqa: SLF001


def _declared_tool_names(system_body: str) -> list[str]:
    """``name(`` tokens declared in a rendered system prompt."""
    return sorted(set(_TOOL_CALL_RE.findall(system_body)))
