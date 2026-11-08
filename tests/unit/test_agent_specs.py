"""Agent-spec contract tests.

The spec is the single definition of an agent: result type, deps type,
templates and retry flags. These tests keep that definition honest without
building a model or calling a provider.
"""

from __future__ import annotations

import typing
from typing import Any, cast

import pytest
from pydantic import BaseModel

from kompend.agents.base import AgentSpec, KompendAgentBase, agent_class, build_agent
from kompend.agents.specs import all_specs, get_spec, registered_names
from kompend.core.templates import get_template_loader


class _ExampleOutput(BaseModel):
    value: str = ""


class _ExampleDeps:
    """Minimal deps shape; the base only needs a class here."""

    source_id: str = "test"


def test_registry_names_match_the_declared_specs() -> None:
    names = registered_names()
    assert names
    assert tuple(spec.name for spec in all_specs().values()) == names
    for name in names:
        assert get_spec(name).name == name


def test_every_registered_spec_materialises_a_typed_class() -> None:
    for name in registered_names():
        spec = get_spec(name)
        cls = cast("type[KompendAgentBase[Any, Any]]", agent_class(spec))
        assert issubclass(cls, KompendAgentBase)
        assert cls.spec is spec


def test_prompt_hooks_are_installed_on_the_materialised_class() -> None:
    for spec in all_specs().values():
        cls = cast("type[KompendAgentBase[Any, Any]]", agent_class(spec))
        if spec.user_prompt is not None:
            assert cls.render_user_prompt is not KompendAgentBase.render_user_prompt
        if spec.system_prompt is not None:
            assert cls.render_system_prompt is not KompendAgentBase.render_system_prompt
        if spec.validate_output is not None:
            assert cls.validate_output is not KompendAgentBase.validate_output


def test_union_result_type_is_supported() -> None:
    spec = get_spec("lint_rule")
    assert len(typing.get_args(spec.result_type)) == 2


def test_spec_types_are_resolved_without_generic_bases() -> None:
    spec = get_spec("extraction")
    agent = build_agent(spec, model_string="ollama/test")

    assert agent.spec is spec
    assert agent.result_type is spec.result_type
    assert agent.deps_type is spec.deps_type
    assert agent._agent is None


def test_materialised_class_is_cached_per_name() -> None:
    spec = get_spec("extraction")
    assert agent_class(spec) is agent_class(spec)
    assert agent_class(spec).__name__ == "ExtractionAgent"


def test_templates_exist_for_every_spec() -> None:
    loader = get_template_loader()
    for spec in all_specs().values():
        for template in (spec.template_name, spec.system_template_name):
            if not template:
                continue
            rendered = loader.get_text(f"{spec.template_prefix}{template}")
            assert rendered.strip()


def test_build_agent_applies_spec_defaults_only_when_unset() -> None:
    spec = AgentSpec(
        name="example_agent",
        result_type=_ExampleOutput,
        deps_type=_ExampleDeps,
        temperature=0.2,
        retries=3,
    )
    agent = build_agent(spec, model_string="ollama/test")
    assert agent.temperature == 0.2
    assert agent.retries == 3

    overridden = build_agent(spec, model_string="ollama/test", temperature=0.9, retries=0)
    assert overridden.temperature == 0.9
    assert overridden.retries == 0


@pytest.mark.parametrize("result_type", [int, list[str], None])
def test_invalid_result_type_is_rejected(result_type: object) -> None:
    with pytest.raises(TypeError):
        AgentSpec(name="bad", result_type=result_type, deps_type=_ExampleDeps)


def test_unknown_spec_name_is_an_error() -> None:
    with pytest.raises(KeyError):
        get_spec("does-not-exist")
