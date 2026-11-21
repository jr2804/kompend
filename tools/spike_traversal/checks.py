"""The three quality spot-checks of the traversal spike (knox-u84v.6).

Each check takes the synthetic document and what one variant produced, and
returns a small dataclass with the numbers. The first two checks are
deterministic; the third asks an LLM judge, one call per planted fact, on a
counter of its own so the variants' measurements stay clean.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

from common import Counter, SyntheticDoc, build_counted_model
from pydantic import BaseModel
from pydantic_ai import Agent

#: The prompt templates this module renders. Text lives here, never in code.
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class JudgeVerdict(BaseModel):
    """The judge's answer for one planted fact."""

    answered: bool
    evidence: str = ""


#: A judge call: question, reference answer and the variant's summaries in,
#: the verdict out. Production judges talk to the model; tests pass a stub.
JudgeFn = Callable[[str, str, str], JudgeVerdict]


@dataclass(frozen=True)
class RootCoverage:
    """Does the root summary name the document's main sections?"""

    root_summarized: bool
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    count: int
    total: int
    fraction: float
    note: str

    def as_dict(self) -> dict:
        """The check as plain data."""
        return {
            "root_summarized": self.root_summarized,
            "matched": list(self.matched),
            "missing": list(self.missing),
            "count": self.count,
            "total": self.total,
            "fraction": round(self.fraction, 4),
            "note": self.note,
        }


@dataclass(frozen=True)
class SkipQuality:
    """Of the skipped branches, how many are boilerplate and how many carry answers?"""

    correct_skips: int
    wrong_skips: int
    neutral_skips: int
    correct_ids: tuple[str, ...]
    wrong_ids: tuple[str, ...]
    neutral_ids: tuple[str, ...]
    note: str

    def as_dict(self) -> dict:
        """The check as plain data."""
        return {
            "correct_skips": self.correct_skips,
            "wrong_skips": self.wrong_skips,
            "neutral_skips": self.neutral_skips,
            "correct_ids": list(self.correct_ids),
            "wrong_ids": list(self.wrong_ids),
            "neutral_ids": list(self.neutral_ids),
            "note": self.note,
        }


@dataclass(frozen=True)
class FactResult:
    """One planted fact and whether the variant's summaries answer it."""

    question: str
    chapter: str
    answered: bool
    evidence: str = ""

    def as_dict(self) -> dict:
        """The fact as plain data."""
        return {
            "question": self.question,
            "chapter": self.chapter,
            "answered": self.answered,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class AnswerCoverage:
    """How many planted facts the variant's summaries answer."""

    answered: int
    total: int
    fraction: float
    missed: tuple[str, ...]
    facts: tuple[FactResult, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        """The check as plain data."""
        return {
            "answered": self.answered,
            "total": self.total,
            "fraction": round(self.fraction, 4),
            "missed": list(self.missed),
            "facts": [fact.as_dict() for fact in self.facts],
        }


def check_root_coverage(doc: SyntheticDoc, summaries: Mapping[str, str]) -> RootCoverage:
    """Count the chapter headings the root summary names.

    Matching is a case-insensitive substring search of each heading in the
    root summary. A variant that skipped or never summarized the root scores
    0 — that is a missing root, not a check that does not apply.
    """
    root_text = summaries.get(doc.tree.root_id, "")
    if not root_text.strip():
        return RootCoverage(
            root_summarized=False,
            matched=(),
            missing=tuple(doc.chapter_headings),
            count=0,
            total=len(doc.chapter_headings),
            fraction=0.0,
            note="the variant never summarized the root: coverage is 0, not n/a",
        )
    lowered = root_text.lower()
    matched = tuple(heading for heading in doc.chapter_headings if heading.lower() in lowered)
    missing = tuple(heading for heading in doc.chapter_headings if heading.lower() not in lowered)
    total = len(doc.chapter_headings)
    return RootCoverage(
        root_summarized=True,
        matched=matched,
        missing=missing,
        count=len(matched),
        total=total,
        fraction=(len(matched) / total) if total else 0.0,
        note="",
    )


def _subtree_ids(doc: SyntheticDoc, node_id: str) -> set[str]:
    """The node and every id below it."""
    seen: set[str] = set()
    stack = [node_id]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        node = doc.tree.nodes.get(current)
        if node is not None:
            stack.extend(node.children_ids)
    return seen


def _ancestor_ids(doc: SyntheticDoc, node_id: str) -> set[str]:
    """Every id above the node, rootwards."""
    seen: set[str] = set()
    node = doc.tree.nodes.get(node_id)
    parent = node.parent_id if node is not None else None
    while parent is not None:
        if parent in seen:
            break
        seen.add(parent)
        parent_node = doc.tree.nodes.get(parent)
        parent = parent_node.parent_id if parent_node is not None else None
    return seen


def check_skip_quality(doc: SyntheticDoc, skips: Mapping[str, str]) -> SkipQuality:
    """Sort the skipped branches into boilerplate, answer-bearing, or neither.

    A skip is wrong when the skipped subtree holds a planted fact — either
    the skipped node carries one itself or it sits above one — because the
    skip loses an answer. A skip is correct when the skipped content is
    boilerplate: the node or one of its ancestors is a boilerplate root.
    Everything else is neutral: a judgement call about real content that
    cost no planted answer. Wrong skips are reported, never averaged away.
    """
    correct: list[str] = []
    wrong: list[str] = []
    neutral: list[str] = []
    for skipped_id in skips:
        subtree = _subtree_ids(doc, skipped_id)
        ancestors = _ancestor_ids(doc, skipped_id)
        if subtree & set(doc.answer_bearing_ids) or ancestors & set(doc.answer_bearing_ids):
            wrong.append(skipped_id)
        elif skipped_id in doc.boilerplate_ids or ancestors & set(doc.boilerplate_ids):
            correct.append(skipped_id)
        else:
            neutral.append(skipped_id)
    return SkipQuality(
        correct_skips=len(correct),
        wrong_skips=len(wrong),
        neutral_skips=len(neutral),
        correct_ids=tuple(sorted(correct)),
        wrong_ids=tuple(sorted(wrong)),
        neutral_ids=tuple(sorted(neutral)),
        note="a skip above or on an answer-bearing branch is wrong: it loses a planted answer",
    )


def render_judge_prompt(question: str, expected_answer: str, summaries_text: str) -> str:
    """Fill the answer-coverage judge template."""
    template = (PROMPTS_DIR / "judge_answer_coverage.md").read_text(encoding="utf-8")
    return Template(template).substitute(
        question=question,
        expected_answer=expected_answer,
        summaries=summaries_text,
    )


def flatten_summaries(doc: SyntheticDoc, summaries: Mapping[str, str]) -> str:
    """All of a variant's summaries as one text, with their headings attached."""
    lines: list[str] = []
    for node_id in sorted(summaries):
        node = doc.tree.nodes.get(node_id)
        heading = " / ".join(node.heading_chain) if node is not None else node_id
        lines.append(f"[{node_id}] {heading}\n{summaries[node_id]}")
    return "\n\n".join(lines)


def llm_judge(counter: Counter) -> JudgeFn:
    """A judge that asks the spike's model, counted on its own counter.

    The counter belongs to the judge, never to a variant: judge calls must
    not pollute the measurements the decision gate compares.
    """
    agent: Agent[None, JudgeVerdict] = Agent(build_counted_model(counter), output_type=JudgeVerdict)

    def judge(question: str, expected_answer: str, summaries_text: str) -> JudgeVerdict:
        """Ask the model whether the summaries answer one fact."""
        prompt = render_judge_prompt(question, expected_answer, summaries_text)
        return agent.run_sync(prompt).output

    return judge


def check_answer_coverage(doc: SyntheticDoc, summaries: Mapping[str, str], judge: JudgeFn) -> AnswerCoverage:
    """Ask the judge, one call per planted fact, whether the summaries answer it."""
    summaries_text = flatten_summaries(doc, summaries)
    results: list[FactResult] = []
    for fact in doc.facts:
        verdict = judge(fact.question, fact.answer, summaries_text)
        results.append(
            FactResult(
                question=fact.question,
                chapter=fact.chapter,
                answered=verdict.answered,
                evidence=verdict.evidence,
            )
        )
    answered = sum(1 for result in results if result.answered)
    total = len(results)
    return AnswerCoverage(
        answered=answered,
        total=total,
        fraction=(answered / total) if total else 0.0,
        missed=tuple(result.question for result in results if not result.answered),
        facts=tuple(results),
    )
