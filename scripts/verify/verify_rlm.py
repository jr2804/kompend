"""Live verification of the RLM integration.

Runs the QueryAgent with RLM off vs. on over a synthetic corpus that is
larger than Ollama's default 2048-token context window, and checks:

- Prompt shape: RLM off inlines the corpus, RLM on swaps in a placeholder
  (rendered without a model call).
- RLM off: the model cannot extract the tail fact from the inlined corpus.
- RLM on: the model finds the fact by code search in the REPL and answers
  correctly.

Not collected by pytest — tests/AGENTS.md forbids real LLM calls in any
test. Run manually:

    uv run python scripts/verify/verify_rlm.py            # default model
    KNOX_TEST_RLM_MODEL=gemma4:e2b uv run python scripts/verify/verify_rlm.py

Exits 0 (SKIP) when Ollama or the model is unavailable; non-zero on
verification failure.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

from knox.agents.deps import QueryDeps
from knox.agents.query import QueryAgent
from knox.core.config._configs import RlmConfig

MODEL = os.environ.get("KNOX_TEST_RLM_MODEL", "hf.co/unsloth/Qwen3.5-2B-MTP-GGUF:UD-Q8_K_XL")
#: Verified on this machine: LFM2.5-8B-A1B ignored the tool, gemma4:e2b
#: emits code as text instead of a tool call, Qwen3.5-4B is too slow.
#: Qwen3.5-2B-MTP calls execute_code and answers correctly (~9 min total).
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
FILLER = "The quick brown fox jumps over the lazy dog. "


def main() -> int:
    if not _ollama_ok():
        print(f"SKIP: Ollama unreachable or model {MODEL!r} not present")
        return 0

    corpus = build_corpus()
    corpus_tokens = len(corpus.split())
    if corpus_tokens <= 4000:
        raise SystemExit(f"corpus too small for truncation: {corpus_tokens} tokens")
    print(f"corpus: {corpus_tokens} approx tokens ({len(corpus)} chars), Ollama default num_ctx=2048")
    sys.stdout.flush()

    # Prompt-shape checks (no model call): off inlines the corpus, on swaps
    # in a placeholder so the conversation stays bounded.
    rendered_off = _rendered_prompt(rlm=None)
    rendered_on = _rendered_prompt(rlm=RlmConfig(enable=True))
    if "<corpus>" not in rendered_off or "<corpus>" in rendered_on:
        raise SystemExit("prompt-shape check failed: rlm-off must inline the context, rlm-on must not")
    print("prompt shapes: rlm-off inlines context, rlm-on uses placeholder -> PASS")

    question = "What is the secret color of document 60? Answer with the color name only."

    # --- Path A: RLM off (context inlined; the model loses the tail fact) ---
    print("running rlm-off path (~4-5 min)...", flush=True)
    answer_off = run_query(question, corpus, rlm=None)
    off_ok = "cerulean" not in answer_off
    print(f"[rlm off ] answer={answer_off[:80]!r} -> {'PASS' if off_ok else 'FAIL'}")

    # --- Path B: RLM on (code search over the full corpus in the REPL) ---
    print("running rlm-on path (~4-5 min)...", flush=True)
    answer_on = run_query(question, corpus, rlm=RlmConfig(enable=True))
    on_ok = "cerulean" in answer_on
    print(f"[rlm on  ] answer={answer_on[:80]!r} -> {'PASS' if on_ok else 'FAIL'}")

    if not (off_ok and on_ok):
        print("\nFAIL: RLM verification failed")
        if on_ok:
            print("  (rlm-off expectation failed — model answered despite truncation?)")
        if not on_ok:
            print("  (rlm-on expectation failed — model did not answer via code search)")
        return 1
    print("\nPASS: RLM keeps the corpus out of the prompt and answers from a corpus larger than the model's context window.")
    return 0


def _rendered_prompt(rlm: RlmConfig | None) -> str:
    """Render the system prompt without any model call — the prompt-shape
    checks are pure Jinja, so they stay fast and deterministic.
    """
    agent = QueryAgent(
        model_string=f"ollama/{MODEL}",
        rlm=rlm,
        rlm_context_field="context",
    )
    return agent.render_system_prompt(QueryDeps(question="q", context="<corpus>", source_id="render"))


def _ollama_ok() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")  # noqa: S310 — local Ollama only
        with urllib.request.urlopen(req, timeout=3) as r:  # noqa: S310 — local Ollama only
            return any(m["name"] == MODEL for m in json.load(r)["models"])
    except Exception:
        return False


def build_corpus(n: int = 80) -> str:
    """Synthetic corpus (~5K+ tokens); the answer fact lives only in the
    last chunk, far beyond Ollama's default 2048-token num_ctx.
    """
    chunks = []
    for i in range(1, n + 1):
        body = FILLER * 5 + "This document describes routine archival notes."
        if i == n:
            body = FILLER * 5 + f"The secret color of document {i} is cerulean."
        chunks.append(f"[{i}] Doc {i}: {body}")
    return "\n".join(chunks)


def run_query(question: str, context: str, rlm: RlmConfig | None) -> str:
    agent = QueryAgent(
        model_string=f"ollama/{MODEL}",
        timeout=600.0,
        rlm=rlm,
        rlm_context_field="context",
    )
    result, _ = agent.run_with_messages(QueryDeps(question=question, context=context, source_id="rlm-verify"))
    return result


if __name__ == "__main__":
    sys.exit(main())
