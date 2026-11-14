"""The relevance and caption prompts stay conservative and specific.

Stage C adds measured image facts to the context; the wording changes here
must keep the conservative contract: doubt answers informative, and the
decorative examples name the concrete classes the cheap filter finds.
"""

from __future__ import annotations

from kompend.core.templates import get_template_loader


def test_the_relevance_system_prompt_names_the_decorative_classes() -> None:
    text = _render("decision/figure_relevance_system.md.j2")

    for term in ("watermark", "separator", "horizontal rule", "border", "solid"):
        assert term in text
    assert "doubt remains" in text
    assert "informative" in text


def test_the_relevance_prompt_keeps_doubt_informative() -> None:
    text = _render(
        "decision/figure_relevance.md.j2",
        context="The surrounding text does not describe the figure.",
    )

    assert "informative" in text
    assert "Anything ambiguous" in text
    assert "watermark" in text
    assert "observations" in text


def test_the_caption_prompt_broadens_the_skip_examples() -> None:
    text = _render("vlm/caption.md.j2", context="")

    for term in ("watermark", "separator", "horizontal rule", "border", "solid", "blank frame"):
        assert term in text
    assert "SKIP" in text


def _render(name: str, **variables: str) -> str:
    return get_template_loader().render(name, **variables)
