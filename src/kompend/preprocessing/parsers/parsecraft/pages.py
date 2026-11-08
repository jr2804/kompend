"""Empty-page accounting and the warnings that say what was lost.

Three different silences can follow a parse — a page that was blank in the
source, a page nobody could read, and a backend that had the capability and
still returned nothing. This module tells them apart, says out loud which
pages fell into each, and quotes parsecraft's own explanation (routing
reason, engine diagnostic, executor signal) instead of guessing at a remedy.

A fourth silence is louder than the other three: a page where every pass
failed contributes zero elements exactly like a blank page, but "empty" and
"failed to parse" need different responses downstream. Those pages get their
own record (`warn_on_exhausted_pages`, plus ``exhausted_pages`` in the
document metadata next to ``degraded_pages``) instead of the empty-page
accounting — ``page_count`` counts the page either way.
"""

from __future__ import annotations

from pathlib import Path

from parsecraft.ir.models import DocumentResult
from parsecraft.pipeline import DEGRADED_EMPTY_OUTPUT_CODE
from parsecraft.pipeline.analysis import AnalysisResult
from parsecraft.routing import RoutingPlan

# parsecraft's page verdict for "this page holds nothing to lose" (no text and
# no images); the planner emits it as ``PageRoute.degradation_code``. Imported
# from upstream so the vocabulary exists exactly once (knox-udom); a pin test
# catches an upstream value change before a vault is parsed with two notions
# of "blank".
from parsecraft.routing.rules import DEGRADED_BLANK_PAGE_CODE as DEGRADED_BLANK_PAGE

from kompend.logging import get_logger
from kompend.preprocessing.parsers.parsecraft.assets import chunk_content

logger = get_logger(__name__)


def warn_on_empty_pages(
    source_file: Path,
    unreadable_pages: list[int],
    blank_pages: list[int],
    plan: RoutingPlan | None,
    document: DocumentResult,
) -> None:
    """Say out loud when pages that *had something to read* contributed nothing.

    Two different losses wear the same face, and only one of them is worth a
    warning:

    * a page with no text and **no images** is blank in the source — a separator
      or a deliberately empty page. Nothing was lost, so nothing is said; a
      warning here reads as a bug report against an OCR setup that cannot help,
      which is exactly how a vault full of ITU-T separators produced a screenful
      of "check the ocr extra" for pages that hold no ink (knox-dizo).
    * a page with no text but **images** is unread: something is there and no
      backend turned it into text. That is the case the warning exists for, and
      it quotes parsecraft's own routing reason ("OCR unavailable (the tesseract
      engine is not installed on this host)") instead of guessing at the remedy.

    Zero content instead of an exception stays possible for either case: a
    backend with no OCR path, or an upstream defect that returns an empty page
    for a scanned PDF (pc-04u). When the engine explains itself on the page —
    docling-rs reports ``docling-ocr-unavailable`` for a half-installed model
    cache — that explanation is quoted instead of being replaced by advice.
    """
    if blank_pages:
        logger.debug(
            "%s: %d blank page(s) (%s) — no text and no images, nothing to extract.",
            source_file.name,
            len(blank_pages),
            ", ".join(str(page) for page in blank_pages),
        )
    if not unreadable_pages:
        return
    logger.warning(
        "%s: %d page(s) hold images but produced no text (%s).%s%s%s",
        source_file.name,
        len(unreadable_pages),
        ", ".join(str(page) for page in unreadable_pages),
        routing_reasons(plan, unreadable_pages),
        page_diagnostics(document, unreadable_pages),
        empty_output_detail(document, unreadable_pages),
    )


def warn_on_exhausted_pages(source_file: Path, exhausted: list[int]) -> None:
    """Say out loud when pages failed on every pass instead of parsing thin.

    One record per file, listing the pages — not one record per page. A
    partial parse still beats nothing, so this never raises; it exists so a
    console ingest shows the gap instead of a clean short parse.
    """
    if not exhausted:
        return
    logger.warning(
        "%s: %d page(s) failed to parse on every pass (%s) — content missing, not empty.",
        source_file.name,
        len(exhausted),
        ", ".join(str(page) for page in exhausted),
    )


def empty_output_detail(document: DocumentResult, page_numbers: list[int]) -> str:
    """What the executor said when a capable backend simply returned nothing.

    ``degraded-empty-output`` is a *quality signal*, not a page diagnostic: the
    executor adds it when an ``ocr_capable`` backend produced no content for a
    page that needs OCR, and it suppresses it on any page whose adapter already
    explained itself, so the two never describe the same page. It has no
    markdown projection, which makes ``document.quality`` the only place it can
    reach a user.
    """
    wanted = set(page_numbers)
    details = {
        f"{signal.detail} ({signal.name})"
        for signal in document.quality
        if signal.page_number in wanted and signal.name == DEGRADED_EMPTY_OUTPUT_CODE and signal.detail
    }
    if not details:
        return ""
    return f" Executor said: {'; '.join(sorted(details))}"


def page_diagnostics(document: DocumentResult, page_numbers: list[int]) -> str:
    """What the engine itself said about these pages, if anything.

    A backend can return an empty page without failing: docling-rs reports a
    partially installed model cache as ``docling-ocr-unavailable`` on the page
    and then converts nothing. That diagnostic is the only place a user learns
    the difference between "this page is unreadable" and "your OCR weights are
    half-downloaded", so it is quoted rather than replaced by advice.
    """
    wanted = set(page_numbers)
    messages = {
        f"{diagnostic.message} ({diagnostic.code})"
        for page in document.pages
        if page.page_number in wanted
        for diagnostic in page.diagnostics
        if diagnostic.message
    }
    if not messages:
        return ""
    return f" Engine said: {'; '.join(sorted(messages))}"


def routing_reasons(plan: RoutingPlan | None, page_numbers: list[int]) -> str:
    """The planner's own words for why these pages produced nothing, if it said any."""
    if plan is None:
        return ""
    reasons = {route.reason.strip() for route in plan.pages if route.page_number in set(page_numbers) and route.reason and route.reason.strip()}
    if not reasons:
        return ""
    return f" Routing said: {'; '.join(sorted(reasons))}"


def classify_empty_pages(empty_pages: list[int], analysis: AnalysisResult, plan: RoutingPlan | None) -> tuple[list[int], list[int]]:
    """Split empty pages into blank-in-the-source and unread.

    The planner owns this decision: ``degradation_code == degraded-blank-page``
    means "nothing to lose" (parsecraft measured 527 s of OCR on one such page),
    and recombining analysis signals here would re-derive a rule it already
    applied. ``image_count`` is the fallback for a route that carries no
    degradation at all — a page with neither text nor images cannot be read by
    any backend. A page with neither a route nor a signal counts as unread:
    silence is not evidence of emptiness.
    """
    codes = {route.page_number: route.degradation_code for route in plan.pages} if plan is not None else {}
    images = {signal.page_number: signal.image_count for signal in analysis.signals}
    blank: list[int] = []
    unreadable: list[int] = []
    for page_number in empty_pages:
        code = codes.get(page_number)
        if code == DEGRADED_BLANK_PAGE or (code is None and images.get(page_number, 1) == 0):
            blank.append(page_number)
        else:
            unreadable.append(page_number)
    return blank, unreadable


def empty_pages(document: DocumentResult) -> list[int]:
    """Page numbers whose blocks carry no content, in document order.

    Content, not block count, decides: a page can hold chunks (headers, an empty
    figure) and still say nothing.
    """
    return [page.page_number for page in document.pages if not any(chunk_content(chunk) for chunk in page.blocks)]
