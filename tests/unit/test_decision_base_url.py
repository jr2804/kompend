"""The decision-model base URL must not duplicate the API version path."""

from __future__ import annotations

import pytest

from kompend.llm.models import _decision_base_url


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("https://ollama.com/v1", "https://ollama.com"),
        ("https://ollama.com/v1/", "https://ollama.com"),
        ("https://openrouter.ai/api/v1", "https://openrouter.ai/api"),
        ("https://api.typesafe.ai", "https://api.typesafe.ai"),
        (None, None),
    ],
)
def test_version_suffix_is_stripped(given: str | None, expected: str | None) -> None:
    assert _decision_base_url(given) == expected
