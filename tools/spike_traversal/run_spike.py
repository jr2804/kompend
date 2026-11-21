"""The traversal-spike runner (knox-u84v.6).

Builds the corpus once, runs variant A and variant B against it — each on
its own counter, so the measurements never mix — checks both with the three
spot-checks, writes ``result.json`` and prints the comparison table with the
decision-gate arithmetic.

Usage:
    mise exec -- uv run python run_spike.py [--variant a|b|both] [--out result.json]

Run from this directory (``tree.py`` imports ``common`` as a top-level
module). Variants that have not landed yet are reported as missing and the
runner stays ready for them.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tree
from checks import check_answer_coverage, check_root_coverage, check_skip_quality, llm_judge
from common import BUDGET_NODES, BUDGET_WINDOWS, MODEL, Counter, Stopwatch, SyntheticDoc, VariantResult, configure_provider

#: The gate the bead sets: navigation wins the calls comparison only when it
#: uses strictly more than this fraction fewer total calls than planner+worker.
CALLS_WIN_FRACTION = 0.25

SPIKE_SEED = 20261121


def load_variant(name: str) -> Any:
    """Import ``variant_a`` / ``variant_b``, or return None when it is missing."""
    try:
        return importlib.import_module(f"variant_{name}")
    except ModuleNotFoundError:
        return None


def same_rebuild(seed: int) -> tuple[SyntheticDoc, SyntheticDoc]:
    """Build the corpus twice and prove the rebuild is identical.

    Both variants walk the same shape: the same node ids, the same headings
    and the same planted facts. Sharing one mutable tree object would let the
    first variant's writes leak into the second variant's run.
    """
    first = tree.build_synthetic_doc(seed=seed)
    second = tree.build_synthetic_doc(seed=seed)
    first_nodes = set(first.tree.nodes)
    second_nodes = set(second.tree.nodes)
    if first.tree.root_id != second.tree.root_id or first_nodes != second_nodes:
        raise ValueError("the corpus rebuild is not identical: node sets differ")
    for node_id in first_nodes:
        old, new = first.tree.nodes[node_id], second.tree.nodes[node_id]
        old_text = old.chunk.text if old.chunk is not None else None
        new_text = new.chunk.text if new.chunk is not None else None
        if old.heading_chain != new.heading_chain or old_text != new_text:
            raise ValueError(f"the corpus rebuild is not identical at node {node_id}")
    if [(fact.question, fact.answer, fact.node_id) for fact in first.facts] != [(fact.question, fact.answer, fact.node_id) for fact in second.facts]:
        raise ValueError("the corpus rebuild is not identical: planted facts differ")
    return first, second


def run_variant(name: str, doc: SyntheticDoc) -> tuple[VariantResult | None, str]:
    """Run one variant on its own counter. Missing modules stay missing."""
    module = load_variant(name)
    if module is None:
        return None, f"variant_{name} has not landed: skipped, runner stays ready"
    counter = Counter()
    with Stopwatch(counter):
        result = module.run(doc, counter)
    if result.variant != name:
        result.notes.append(f"module reported variant {result.variant!r}, expected {name!r}")
        result.variant = name
    return result, ""


def check_variant(doc: SyntheticDoc, result: VariantResult, judge_counter: Counter) -> dict:
    """The three spot-checks for one variant's output."""
    judge = llm_judge(judge_counter)
    root = check_root_coverage(doc, result.summaries)
    skip = check_skip_quality(doc, result.skips)
    answer = check_answer_coverage(doc, result.summaries, judge)
    return {
        "root_coverage": root.as_dict(),
        "skip_quality": skip.as_dict(),
        "answer_coverage": answer.as_dict(),
    }


def calls_verdict(requests_a: int, requests_b: int) -> tuple[bool, str]:
    """Apply the calls half of the gate: B must use strictly more than 25 % fewer calls."""
    if requests_a <= 0:
        return False, "no baseline: variant A made no calls, so no relative difference exists"
    saved = (requests_a - requests_b) / requests_a
    if saved > CALLS_WIN_FRACTION:
        return True, f"B uses {saved:.1%} fewer calls than A — strictly more than {CALLS_WIN_FRACTION:.0%}"
    return False, f"B uses {saved:.1%} fewer calls than A — not more than {CALLS_WIN_FRACTION:.0%}"


def quality_verdict(checks_a: dict, checks_b: dict) -> tuple[bool, str]:
    """Apply the quality half of the gate: B clearly better means better on one check, worse on none."""
    wins = 0
    losses = 0
    better_on: list[str] = []
    worse_on: list[str] = []
    answer_a = checks_a["answer_coverage"]["answered"]
    answer_b = checks_b["answer_coverage"]["answered"]
    if answer_b > answer_a:
        wins += 1
        better_on.append(f"answer coverage {answer_b} > {answer_a}")
    elif answer_b < answer_a:
        losses += 1
        worse_on.append(f"answer coverage {answer_b} < {answer_a}")
    root_a = checks_a["root_coverage"]["fraction"]
    root_b = checks_b["root_coverage"]["fraction"]
    if root_b > root_a:
        wins += 1
        better_on.append(f"root coverage {root_b:.2f} > {root_a:.2f}")
    elif root_b < root_a:
        losses += 1
        worse_on.append(f"root coverage {root_b:.2f} < {root_a:.2f}")
    wrong_a = checks_a["skip_quality"]["wrong_skips"]
    wrong_b = checks_b["skip_quality"]["wrong_skips"]
    if wrong_b < wrong_a:
        wins += 1
        better_on.append(f"wrong skips {wrong_b} < {wrong_a}")
    elif wrong_b > wrong_a:
        losses += 1
        worse_on.append(f"wrong skips {wrong_b} > {wrong_a}")
    if wins > 0 and losses == 0:
        return True, "B clearly better: " + "; ".join(better_on)
    if losses > 0:
        return False, "B not clearly better: worse on " + "; ".join(worse_on)
    return False, "B not clearly better: level with A on every spot-check"


def render_table(results: dict[str, VariantResult | None], checked: dict[str, dict]) -> str:
    """The comparison table, one row per measurement, one column per variant."""
    rows: list[tuple[str, str, str]] = []

    def row(label: str, fmt_a: str, fmt_b: str) -> None:
        rows.append((label, fmt_a, fmt_b))

    def fmt(value: object) -> str:
        return str(value)

    table: dict[str, dict[str, str]] = {}
    for name in ("a", "b"):
        result = results.get(name)
        column: dict[str, str] = {}
        if result is None:
            column = dict.fromkeys(
                (
                    "requests",
                    "decision",
                    "summarize",
                    "other",
                    "windows",
                    "prompt",
                    "completion",
                    "peak",
                    "wall",
                    "summaries",
                    "skips",
                    "root",
                    "wrong",
                    "answered",
                ),
                "—",
            )
        else:
            counter = result.counter.as_dict()
            check = checked[name]
            column = {
                "requests": fmt(counter["requests"]),
                "decision": fmt(counter["decision_requests"]),
                "summarize": fmt(counter["summarize_requests"]),
                "other": fmt(counter["other_requests"]),
                "windows": fmt(counter["windows"]),
                "prompt": fmt(counter["prompt_tokens"]),
                "completion": fmt(counter["completion_tokens"]),
                "peak": fmt(counter["peak_prompt_tokens"]),
                "wall": fmt(counter["wall_time_s"]),
                "summaries": fmt(len(result.summaries)),
                "skips": fmt(len(result.skips)),
                "root": f"{check['root_coverage']['count']}/{check['root_coverage']['total']}",
                "wrong": fmt(check["skip_quality"]["wrong_skips"]),
                "answered": f"{check['answer_coverage']['answered']}/{check['answer_coverage']['total']}",
            }
        table[name] = column
    labels = [
        ("requests", "total calls"),
        ("decision", "decision calls"),
        ("summarize", "summarize calls"),
        ("other", "other calls"),
        ("windows", "windows"),
        ("prompt", "prompt tokens"),
        ("completion", "completion tokens"),
        ("peak", "peak prompt tokens"),
        ("wall", "wall time s"),
        ("summaries", "summaries"),
        ("skips", "skips"),
        ("root", "root coverage"),
        ("wrong", "wrong skips"),
        ("answered", "facts answered"),
    ]
    for key, label in labels:
        row(label, table["a"][key], table["b"][key])
    width = max(len(label) for label, _, _ in rows)
    lines = [f"{'measure':<{width}}  A  B", "-" * (width + 6)]
    lines.extend(f"{label:<{width}}  {fmt_a}  {fmt_b}" for label, fmt_a, fmt_b in rows)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Build the corpus, run the variants, check them, record the verdict."""
    parser = argparse.ArgumentParser(description="Traversal-spike runner (knox-u84v.6)")
    parser.add_argument("--variant", choices=("a", "b", "both"), default="both")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "result.json"))
    args = parser.parse_args(argv)

    names = ["a", "b"] if args.variant == "both" else [args.variant]
    configure_provider()

    doc_a, doc_b = same_rebuild(SPIKE_SEED)
    docs = {"a": doc_a, "b": doc_b}
    results: dict[str, VariantResult | None] = {}
    missing: list[str] = []
    for name in names:
        result, note = run_variant(name, docs[name])
        results[name] = result
        if result is None:
            missing.append(note)

    judge_counter = Counter()
    checked: dict[str, dict] = {}
    for name in names:
        result = results[name]
        checked[name] = check_variant(docs[name], result, judge_counter) if result is not None else {}

    payload: dict = {
        "corpus": {
            "seed": SPIKE_SEED,
            "nodes": len(doc_a.tree.nodes),
            "facts": len(doc_a.facts),
            "chapters": len(doc_a.chapter_headings),
        },
        "model": MODEL,
        "budget": {"nodes": BUDGET_NODES, "windows": BUDGET_WINDOWS},
        "variants": {},
        "judge": judge_counter.as_dict(),
    }
    for name in names:
        result = results[name]
        payload["variants"][name] = {**result.as_dict(), "checks": checked[name]} if result is not None else {"status": "missing"}

    verdict_lines: list[str] = []
    gate: dict = {
        "rule": "navigation wins iff it uses strictly more than 25 % fewer total calls, "
        "or is clearly better on the spot-checks (better on one, worse on none); "
        "otherwise planner + worker stays"
    }
    if all(results.get(name) is not None for name in ("a", "b")) and args.variant == "both":
        result_a = results["a"]
        result_b = results["b"]
        if result_a is None or result_b is None:
            raise ValueError("need both variants to decide the gate")
        calls_win, calls_why = calls_verdict(result_a.counter.requests, result_b.counter.requests)
        quality_win, quality_why = quality_verdict(checked["a"], checked["b"])
        navigation_wins = calls_win or quality_win
        verdict = "navigation (B) wins" if navigation_wins else "planner + worker (A) stays"
        gate.update(
            {
                "total_calls_a": result_a.counter.requests,
                "total_calls_b": result_b.counter.requests,
                "calls_win": calls_win,
                "calls_why": calls_why,
                "quality_win": quality_win,
                "quality_why": quality_why,
                "verdict": verdict,
            }
        )
        verdict_lines.append(f"calls: {calls_why}")
        verdict_lines.append(f"quality: {quality_why}")
        verdict_lines.append(f"verdict: {verdict}")
    else:
        gate.update({"verdict": "undecided: need both variants"})
        verdict_lines.append("verdict: undecided — need both variants")
        verdict_lines.extend(missing)
    payload["gate"] = gate

    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"corpus: {payload['corpus']['nodes']} nodes, {payload['corpus']['facts']} facts, seed {SPIKE_SEED}")
    print(f"model: {MODEL} — budget: {BUDGET_NODES} nodes / {BUDGET_WINDOWS} windows")
    print()
    print(render_table(results, checked))
    print()
    print(f"gate rule: {gate['rule']}")
    for line in verdict_lines:
        print(line)
    print(f"judge calls (own counter, excluded from the gate): {judge_counter.requests}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
