from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module

from kompend.core.config._configs import DecisionConfig
from kompend.search.retriever import TOKEN_SPLIT_RE

_GREETING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(hi|hello|hey|heya|howdy|yo)\b", re.IGNORECASE),
    re.compile(r"\b(good\s*(morning|afternoon|evening|day))\b", re.IGNORECASE),
    re.compile(r"^(sup|what's up|wassup)\b", re.IGNORECASE),
]

_GRATITUDE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(thanks|thank you|thx|ty|appreciate it|thankful)\b", re.IGNORECASE),
]

_HELP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(what can you do|help|commands?|how (do|can) you (work|help))\b", re.IGNORECASE),
    re.compile(r"^(what are|list)\s+(your )?(commands|capabilities)\b", re.IGNORECASE),
]

_OFFTOPIC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\btell me a joke\b", re.IGNORECASE),
    re.compile(r"\bhow'?s the weather\b", re.IGNORECASE),
    re.compile(r"\bwhat time is it\b", re.IGNORECASE),
    re.compile(r"\bplay (music|a song)\b", re.IGNORECASE),
    re.compile(r"\bwho (are you|made you|created you)\b", re.IGNORECASE),
]

_GREETING_RESPONSE: str = (
    "Hello! I can answer questions about your compiled wiki. Try asking something like "
    "'what is flash attention?' or run `kompend query --help` for more options."
)
_GRATITUDE_RESPONSE: str = "You're welcome! If you have more questions about your wiki, just ask."
_HELP_RESPONSE: str = 'I can search your compiled wiki for relevant information. Try: `kompend query "your question"` or `kompend query --help`.'
_OFFTOPIC_RESPONSE: str = (
    "I'm a knowledge base assistant. I can answer questions about your compiled wiki. Try asking something about the documents you've compiled."
)


_STOP: frozenset[str] = frozenset(
    {
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "had",
        "her",
        "was",
        "one",
        "our",
        "out",
        "has",
        "have",
        "been",
        "some",
        "them",
        "than",
        "that",
        "this",
        "very",
        "just",
        "also",
        "from",
        "with",
        "what",
        "when",
        "where",
        "which",
        "will",
        "your",
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank",
        "good",
        "morning",
        "afternoon",
        "evening",
        "help",
        "please",
        "appreciate",
        "welcome",
        "howdy",
    }
)


class QueryIntent(StrEnum):
    GREETING = "greeting"
    GRATITUDE = "gratitude"
    HELP = "help"
    OFFTOPIC = "offtopic"
    KNOWLEDGE = "knowledge"

    @classmethod
    @functools.cache
    def get_classifiers(cls) -> tuple[ClassifierRule, ...]:
        """Return classification rules in priority order."""
        return (
            ClassifierRule(cls.OFFTOPIC, tuple(_OFFTOPIC_PATTERNS), _OFFTOPIC_RESPONSE, False),
            ClassifierRule(cls.GREETING, tuple(_GREETING_PATTERNS), _GREETING_RESPONSE, True),
            ClassifierRule(cls.GRATITUDE, tuple(_GRATITUDE_PATTERNS), _GRATITUDE_RESPONSE, False),
            ClassifierRule(cls.HELP, tuple(_HELP_PATTERNS), _HELP_RESPONSE, True),
        )


@dataclass(frozen=True)
class ClassifierRule:
    """A single classification rule with intent, patterns, response, and match mode."""

    intent: QueryIntent
    patterns: tuple[re.Pattern[str], ...]
    response: str
    match_start: bool


@dataclass(frozen=True)
class ClassificationResult:
    intent: QueryIntent
    response: str | None


def classify(
    query: str,
    min_length: int = 2,
    *,
    decision: DecisionConfig | None = None,
) -> ClassificationResult:
    """Classify *query* into the assistant's intents.

    The regex rules run first: they are free and exact on the obvious cases
    (a bare "hello", "tell me a joke"). Only a query they leave as knowledge
    reaches *decision*, the optional decision-model sidecar, which recognises
    small talk the patterns miss. A disabled or failing sidecar keeps the
    deterministic answer, so the classifier never depends on a model call.
    """
    lower: str = query.strip().lower()

    if len(lower) < min_length:
        return ClassificationResult(QueryIntent.KNOWLEDGE, response=None)

    for rule in QueryIntent.get_classifiers():
        if rule.intent == QueryIntent.OFFTOPIC:
            for p in rule.patterns:
                if p.search(lower):
                    return ClassificationResult(rule.intent, response=rule.response)
        elif _has_substance(lower):
            break
        else:
            for p in rule.patterns:
                matched = p.match if rule.match_start else p.search
                if matched(lower):
                    return ClassificationResult(rule.intent, response=rule.response)

    verdict = _decide_intent(query, decision)
    if verdict is not None:
        return ClassificationResult(verdict, response=intent_response(verdict))

    return ClassificationResult(QueryIntent.KNOWLEDGE, response=None)


def intent_response(intent: QueryIntent) -> str | None:
    """The canned reply for *intent*, or ``None`` when retrieval should run."""
    return next((rule.response for rule in QueryIntent.get_classifiers() if rule.intent == intent), None)


def _decide_intent(query: str, decision: DecisionConfig | None) -> QueryIntent | None:
    """Consult the decision sidecar, if one is configured.

    The agents layer pulls pydantic-ai and the provider SDKs in, and the CLI
    imports this module on every query, so the import is runtime-only and
    happens only when a decision model is actually enabled.
    """
    if decision is None or not decision.enabled:
        return None
    decide = import_module("kompend.agents.decision").decide_query_intent
    verdict = decide(decision, query=query)
    if verdict is None:
        return None
    return QueryIntent(verdict.value)


def _has_substance(query: str) -> bool:
    tokens: list[str] = TOKEN_SPLIT_RE.split(query.strip().lower())
    content_tokens: list[str] = [t for t in tokens if len(t) > 2 and t not in _STOP]
    return len(content_tokens) >= 2
