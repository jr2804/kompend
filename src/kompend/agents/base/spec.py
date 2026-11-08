"""Declarative agent specs — data-only agent definitions.

An :class:`AgentSpec` names one agent and carries everything that used to be
class-level configuration on a ``KompendAgentBase`` subclass: the result and
deps types, the prompt templates, the delta-retry opt-in and the model
builder. :func:`build_agent` materialises the spec into a typed
``KompendAgentBase`` subclass, so call sites construct agents from data
instead of hand-wiring a class.

The spec is a *kompend* spec, not PydanticAI's ``AgentSpec``: kompend's
prompts are Jinja2 templates rendered from typed deps, which PydanticAI's
``instructions`` string/template cannot express. The concepts map one to
one; the runtime still builds a real PydanticAI :class:`Agent`.

Specs are pure data. Prompt-rendering overrides that cannot be expressed as a
template name (the search agents receive an already-rendered prompt, the
doctor renders Markdown from the report) attach callables to the spec; the
materialised class delegates to them unchanged.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from pydantic import BaseModel
from pydantic_ai.agent.abstract import AgentRetries
from pydantic_ai.models import Model

from kompend.agents.base.deps import AgentDeps

#: Names that a spec must not reuse: the generated class would collide with
#: an existing hand-written class and silently change its identity.
_RESERVED_NAMES: frozenset[str] = frozenset({"kompend_agent_base", "spec_agent"})


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """One agent's declarative definition.

    ``result_type`` is the Pydantic model (or ``str``) the agent must return.
    ``deps_type`` is the dataclass passed to ``run``. Template names are
    relative to ``template_prefix`` under the templates root.
    """

    name: str
    result_type: Any
    deps_type: type[AgentDeps]
    template_name: str = ""
    system_template_name: str = ""
    template_prefix: str = "compilation/"
    supports_delta_retry: bool = False
    temperature: float | None = None
    use_native_output: bool = False
    retries: int | AgentRetries | None = None
    model_builder: Callable[[str], Model] | None = None
    #: Optional prompt hooks for agents whose prompt is computed, not
    #: templated. The hook receives the agent instance and the deps, so it can
    #: read ``agent.rlm`` / ``agent.skills`` / ``agent._template_loader`` the
    #: way the class method it replaces did.
    user_prompt: Callable[[Any, Any], str] | None = None
    system_prompt: Callable[[Any, Any], str] | None = None
    #: Optional post-run validation hook (``ctx, output``), installed as a
    #: static method exactly like the hand-written override.
    validate_output: Callable[[Any, Any], Any] | None = None
    #: Free-form provenance used by the validation test and docs.
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AgentSpec.name must be non-empty")
        if self.name in _RESERVED_NAMES:
            raise ValueError(f"AgentSpec.name {self.name!r} is reserved")
        if not isinstance(self.deps_type, type):
            raise TypeError(f"AgentSpec.deps_type must be a class, got {self.deps_type!r}")
        _validate_result_type(self.name, self.result_type)


def _validate_result_type(name: str, result_type: Any) -> None:
    """Reject output types the base class cannot hand to PydanticAI."""
    if result_type is str or result_type is dict:
        return
    if isinstance(result_type, type) and issubclass(result_type, BaseModel):
        return
    origin = typing.get_origin(result_type)
    if origin is dict:
        return
    if origin is typing.Union or origin is types.UnionType:
        args = typing.get_args(result_type)
        if args and all(_is_supported_result(part) for part in args):
            return
    raise TypeError(f"AgentSpec {name!r}: result_type must be str, dict[...], a BaseModel subclass or a union of those")


def _is_supported_result(result_type: Any) -> bool:
    """One arm of a union result type."""
    if result_type is str or result_type is dict:
        return True
    if isinstance(result_type, type) and issubclass(result_type, BaseModel):
        return True
    return typing.get_origin(result_type) is dict


def _class_name(spec_name: str) -> str:
    """``extraction_reduce`` -> ``ExtractionReduceAgent``."""
    parts = [part for part in spec_name.replace("-", "_").split("_") if part]
    return "".join(part.capitalize() for part in parts) + "Agent"


_SPEC_CLASSES: dict[str, type] = {}


def agent_class(spec: AgentSpec) -> type:
    """Materialise (and cache) the ``KompendAgentBase`` subclass for *spec*."""
    cached = _SPEC_CLASSES.get(spec.name)
    if cached is not None:
        return cached

    # Imported by name: agent.py imports AgentSpec, so a module-level import
    # would close the cycle spec -> agent -> spec.
    KompendAgentBase = import_module("kompend.agents.base.agent").KompendAgentBase

    attrs: dict[str, Any] = {
        "__doc__": f"Spec-backed agent for {spec.name!r}.",
        "__module__": __name__,
        "spec": spec,
        "template_name": spec.template_name,
        "system_template_name": spec.system_template_name,
        "template_prefix": spec.template_prefix,
        "supports_delta_retry": spec.supports_delta_retry,
        "model_builder": staticmethod(spec.model_builder) if spec.model_builder is not None else None,
    }
    if spec.temperature is not None:
        attrs["temperature"] = spec.temperature

    cls = type(_class_name(spec.name), (KompendAgentBase,), attrs)
    _SPEC_CLASSES[spec.name] = cls
    return cls


def install_hooks(cls: type, spec: AgentSpec) -> None:
    """Install a spec's prompt/validation hooks on its materialised class.

    The wrappers keep the base class's method signature (``self, deps``) while
    the spec hook receives ``(agent, deps)`` — a hook can read agent state
    without reaching back through the class.
    """
    target: Any = cls
    if spec.user_prompt is not None:
        hook = spec.user_prompt

        def render_user_prompt(self: Any, deps: Any, _hook: Any = hook) -> str:
            return _hook(self, deps)

        target.render_user_prompt = render_user_prompt
    if spec.system_prompt is not None:
        system_hook = spec.system_prompt

        def render_system_prompt(self: Any, deps: Any, _hook: Any = system_hook) -> str:
            return _hook(self, deps)

        target.render_system_prompt = render_system_prompt
    if spec.validate_output is not None:
        target.validate_output = staticmethod(spec.validate_output)


def build_agent(spec: AgentSpec | str, *, model_string: str, **kwargs: Any) -> Any:
    """Build the agent described by *spec* (or a registered spec name).

    Scalar defaults declared on the spec (``temperature``,
    ``use_native_output``, ``retries``) are applied only when the call site
    did not pass them explicitly, so a site keeps its say while the spec
    carries the default.
    """
    if isinstance(spec, str):
        spec = import_module("kompend.agents.specs").get_spec(spec)
    cls = agent_class(spec)
    if spec.temperature is not None:
        kwargs.setdefault("temperature", spec.temperature)
    if spec.use_native_output:
        kwargs.setdefault("use_native_output", True)
    if spec.retries is not None:
        kwargs.setdefault("retries", spec.retries)
    return cls(model_string=model_string, **kwargs)


def spec_names() -> tuple[str, ...]:
    """Registered spec names, without importing the agent modules."""
    return import_module("kompend.agents.specs").registered_names()
