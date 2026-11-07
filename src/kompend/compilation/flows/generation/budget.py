"""Request and token accounting for the agentic generation pass.

Every reserved payload of a generation request is named here and measured
from the templates and variable content; the shared planner owns the window,
the answer reserve and the safety margin.
"""

from __future__ import annotations

from kompend.compilation.two_phase import TwoPhaseRuntime
from kompend.core.templates import get_template_loader
from kompend.llm.request_budget import RequestBudget
from kompend.logging import get_logger

#: Fixed per-turn agentic conversation framing (tool call + result parts)
#: beyond the content itself. Sized for a full read_segment + fetch_concepts
#: round and a follow-up fetch round (models often loop several rounds
#: before answering).
_AGENTIC_TOOL_FRAMING_TOKENS = 900
#: Per fetch_concepts round: assistant tool call + result framing.
_AGENTIC_FETCH_ROUND_TOKENS = 600
#: Per list_known_titles round: assistant tool call + result framing, on top
#: of the title window itself.
_AGENTIC_TITLES_ROUND_TOKENS = 600

#: Share of the context window the known-titles payload may occupy in total,
#: split evenly between the head inlined in the prompt and one page served by
#: ``list_known_titles()``. The titles are the one reserved payload that grows
#: with the corpus (every concept name plus the previous compile's aliases;
#: observed: 8156 titles ≈ 47K tokens against a 64K window, whose segment
#: budget collapsed to the 512-token floor and aborted the run). They are
#: *windowed*, never truncated: half the share is in the prompt, the other
#: half is the tool page, so every title stays reachable to the model while
#: the segment budget is bounded at planning time.
_KNOWN_TITLES_MAX_FRACTION = 0.10

#: How many provider-named window limits one agentic generation run may
#: re-plan against. A rejection naming a lower limit lowers the window and
#: re-segments the sources; the loop ends as soon as the provider repeats a
#: limit kompend already plans against, and this cap stops a provider that
#: keeps naming new ones from re-segmenting the whole corpus forever.
_MAX_CONTEXT_CLAMPS: int = 3


logger = get_logger(__name__)
_loader = get_template_loader()


def _agentic_budget(
    compiler: TwoPhaseRuntime,
    known_titles_prompt: list[str],
    titles_page_chars: int,
    pinned_clean: list[str],
) -> RequestBudget:
    """Fixed per-request cost of an agentic generation turn.

    Everything that lands in the request that is NOT the segment content:
    system template + rendered user prompt (template body + the inline
    known_titles head + pinned_facts blocks + categories) + the
    read_segment/fetch_concepts tool-call framing + one fetch result + one
    list_known_titles round (its framing and the title window it serves).

    Measured from the actual templates and variable payloads (same
    chars-per-token heuristic used for segment sizing), never hard-coded
    estimates. The window, the answer reserve and the safety margin are the
    planner's business — this function only names the payloads.
    """
    cpt = compiler._config.llm.default_chars_per_token
    categories = compiler._config.extraction_categories
    reserved = {
        "system": _token_estimate(_loader.get_body("compilation/generation_agentic_system.md.j2"), cpt),
        "prompt": _token_estimate(_loader.get_body("compilation/generation_agentic_prompt.md.j2"), cpt),
        "known_titles": _token_estimate("\n".join(known_titles_prompt), cpt),
        "pinned_facts": _token_estimate("\n".join(pinned_clean), cpt),
        "categories": _token_estimate("\n".join(f"{k}: {v}" for k, v in categories.items()), cpt),
        "tool_framing": _AGENTIC_TOOL_FRAMING_TOKENS,
        "fetch_round": _AGENTIC_FETCH_ROUND_TOKENS,
    }
    if titles_page_chars > 0:
        # The model may page through the titles the prompt does not inline:
        # reserve that round (framing + the window it returns).
        reserved["titles_round"] = _token_estimate_chars(titles_page_chars, cpt) + _AGENTIC_TITLES_ROUND_TOKENS
    budget = RequestBudget(window=compiler._context_window, reserved=reserved)
    logger.info("TwoPhase: agentic budget (%s)", budget.report())
    return budget


def _token_estimate(text: str, chars_per_token: float) -> int:
    return max(1, int(round(len(text) / chars_per_token)))


def _token_estimate_chars(chars: int, chars_per_token: float) -> int:
    return max(1, int(round(chars / chars_per_token)))


def _known_titles_windows(compiler: TwoPhaseRuntime, titles: list[str]) -> tuple[list[str], int]:
    """Split the known-titles payload into windows instead of cropping it.

    The titles are the one reserved payload that grows with the corpus: this
    compile's concept names plus the previous compile's learned aliases.
    Unbounded they consumed the whole window (8156 titles ≈ 47K tokens
    against a 64K window), leaving no room for segment content — every
    segment then tripped the request-fit guard and the run aborted after all
    upstream work.

    The share of the window the titles may use is split in two: the head
    inlined in the prompt and one page served by ``list_known_titles()``.
    No title is dropped — the tool reaches the rest of the list with the
    offset the template names. Titles keep their order (this compile's
    concepts first, previous-compile aliases last), so the most relevant
    ones are already in the prompt.

    Returns the inline head and the character size of one tool page — 0 when
    the whole list fits inline, so nothing is reserved for a round the model
    never needs.
    """
    cpt = compiler._config.llm.default_chars_per_token
    share_tokens = max(1, int(compiler._context_window.max_input * _KNOWN_TITLES_MAX_FRACTION / 2))
    share_chars = int(share_tokens * cpt)
    head: list[str] = []
    used = 0
    for title in titles:
        cost = len(title) + 1  # newline separator
        if head and used + cost > share_chars:
            break
        head.append(title)
        used += cost
    if len(head) >= len(titles):
        return head, 0
    logger.info(
        "TwoPhase: known_titles windowed — %d of %d titles inline, the rest served by list_known_titles()"
        " in pages of %d chars (%.0f%% of the %d-token window total)",
        len(head),
        len(titles),
        share_chars,
        _KNOWN_TITLES_MAX_FRACTION * 100,
        compiler._context_window.max_input,
    )
    return head, share_chars
