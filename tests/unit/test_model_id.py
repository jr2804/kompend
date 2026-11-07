"""Pins for provider/model string parsing: ``kompend.llm.model_id`` and its call sites.

Every consumer of a model string must answer "which provider is this model"
by the same rule (first ``/`` or ``:`` in the string, never position 0; bare
model → no provider). The module pins the union rule; the call-site tests
pin the recorded disagreements this module resolved (worker clipping and
debug slugs missed the ``provider:model`` colon form; the doctor mis-split
``ollama:hf.co/...``; see ``.agents/REFACTORING.md``, the model_id loop).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from kompend.core.env import EnvVar
from kompend.llm import model_slug, provider_id_of
from kompend.llm.debug import save_debug_prompt
from kompend.llm.models import _split
from kompend.llm.user_providers import effective_workers, set_user_providers
from kompend.rag.vector_store import to_pydantic_ai_embedding_string

#: Filename layout of a debug dump: ``llm_<YYYYmmdd>_<HHMMSS>_<ffffff>_<slug>.json``.
_DEBUG_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^llm_\d{8}_\d{6}_\d{6}_(.*)\.json$")

_ALL_FORMS: Final[tuple[str, ...]] = (
    "ollama/llama3",
    "ollama:qwen3:latest",
    "ollama/hf.co/Qwen/Qwen3-4B",
    "ollama:hf.co/Qwen/Qwen3-4B",
    "hf.co/Qwen/Qwen3-4B",
    "llama3",
    "/llama3",
    ":weird",
)

_SEPARATED_FORMS: Final[tuple[str, ...]] = (
    "ollama/llama3",
    "ollama:qwen3:latest",
    "ollama/hf.co/Qwen/Qwen3-4B",
    "ollama:hf.co/Qwen/Qwen3-4B",
    "hf.co/Qwen/Qwen3-4B",
)


class TestModelId:
    """The union rule in ``kompend.llm.model_id`` itself."""

    @pytest.mark.parametrize(
        ("model_string", "expected"),
        [
            ("ollama/llama3", "ollama"),
            ("ollama:qwen3:latest", "ollama"),
            ("ollama/hf.co/Qwen/Qwen3-4B", "ollama"),
            ("ollama:hf.co/Qwen/Qwen3-4B", "ollama"),
            ("hf.co/Qwen/Qwen3-4B", "hf.co"),
            ("llama3", None),
            ("/llama3", None),
            (":weird", None),
        ],
    )
    def test_provider_id_of_forms(self, model_string: str, expected: str | None) -> None:
        assert provider_id_of(model_string) == expected

    @pytest.mark.parametrize(
        ("model_string", "expected"),
        [
            ("ollama/llama3", "llama3"),
            ("ollama:qwen3:latest", "qwen3:latest"),
            ("ollama/hf.co/Qwen/Qwen3-4B", "hf.co/Qwen/Qwen3-4B"),
            ("ollama:hf.co/Qwen/Qwen3-4B", "hf.co/Qwen/Qwen3-4B"),
            ("hf.co/Qwen/Qwen3-4B", "Qwen/Qwen3-4B"),
            ("llama3", "llama3"),
            ("/llama3", "/llama3"),
            (":weird", ":weird"),
        ],
    )
    def test_model_slug_forms(self, model_string: str, expected: str) -> None:
        assert model_slug(model_string) == expected

    @pytest.mark.parametrize("model_string", _ALL_FORMS)
    def test_slug_is_the_verbatim_remainder(self, model_string: str) -> None:
        """The model part is exactly the remainder after the original separator."""
        provider = provider_id_of(model_string)
        if provider is None:
            assert model_slug(model_string) == model_string
        else:
            assert model_slug(model_string) == model_string[len(provider) + 1 :]

    @pytest.mark.parametrize("model_string", _SEPARATED_FORMS)
    def test_agrees_with_build_path_split(self, model_string: str) -> None:
        assert (provider_id_of(model_string), model_slug(model_string)) == _split(model_string)


class TestEffectiveWorkersColonForm:
    """``effective_workers`` must clip for both spellings (recorded disagreement D1)."""

    def test_clips_colon_and_slash_spellings_identically(self) -> None:
        set_user_providers({"ollama": {"max_connections": 2}})
        assert effective_workers("ollama:qwen3:latest", 8) == 2
        assert effective_workers("ollama/qwen3:latest", 8) == 2

    def test_bare_and_unknown_providers_untouched(self) -> None:
        set_user_providers({"ollama": {"max_connections": 2}})
        assert effective_workers("llama3", 8) == 8
        assert effective_workers("mistral/x", 8) == 8

    @pytest.fixture(autouse=True)
    def _clean_user_providers(self) -> Iterator[None]:
        set_user_providers({})
        yield
        set_user_providers({})


class _FakeAgent:
    """The minimal agent surface ``save_debug_prompt`` reads."""

    stage = "test"
    current_source_id = "src"
    _last_debug_file: Path | None = None


class TestDebugSlugSpellingIndependence:
    """The debug slug must strip the provider for both spellings (recorded disagreement D2)."""

    def test_same_slug_for_colon_and_slash_spellings(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        slash_dir = str(tmp_path / "slash")
        colon_dir = str(tmp_path / "colon")
        Path(slash_dir).mkdir()
        Path(colon_dir).mkdir()
        assert self._dump_slug(monkeypatch, slash_dir, "ollama/qwen3:latest") == "qwen3latest"
        assert self._dump_slug(monkeypatch, colon_dir, "ollama:qwen3:latest") == "qwen3latest"

    def _dump_slug(self, monkeypatch: pytest.MonkeyPatch, debug_dir: str, model_string: str) -> str:
        monkeypatch.setenv(EnvVar.DEBUG_LLM_DIR, debug_dir)
        save_debug_prompt(
            _FakeAgent(),
            model_string,
            [{"role": "user", "content": "hi"}],
            {},
            "{}",
            None,
        )
        files = list(Path(debug_dir).iterdir())
        assert len(files) == 1
        match = _DEBUG_NAME_PATTERN.match(files[0].name)
        assert match is not None, f"unexpected debug filename: {files[0].name}"
        return match.group(1)


class TestEmbeddingStringTranslation:
    """``to_pydantic_ai_embedding_string`` accepts exactly the known-provider prefixes."""

    @pytest.mark.parametrize(
        ("model_string", "expected"),
        [
            ("ollama:granite-embedding:278m", "ollama:granite-embedding:278m"),
            ("openai/text-embedding-3-small", "openai:text-embedding-3-small"),
            ("ollama/nomic-embed-text", "ollama:nomic-embed-text"),
            ("ollama/hf.co/Qwen/Qwen3-Embedding-0.6B", "ollama:hf.co/Qwen/Qwen3-Embedding-0.6B"),
            ("ollama:hf.co/Qwen/Qwen3-Embedding-0.6B", "ollama:hf.co/Qwen/Qwen3-Embedding-0.6B"),
            ("openai/gpt-4o:2024-11-20", "openai:gpt-4o:2024-11-20"),
        ],
    )
    def test_translates_known_providers(self, model_string: str, expected: str) -> None:
        assert to_pydantic_ai_embedding_string(model_string) == expected

    @pytest.mark.parametrize(
        "model_string",
        ["granite-embedding:278m", "llama3", "hf.co/Qwen/Qwen3-4B", "unknown/model"],
    )
    def test_rejects_unknown_or_bare_providers(self, model_string: str) -> None:
        with pytest.raises(ValueError, match="no known provider prefix"):
            to_pydantic_ai_embedding_string(model_string)
