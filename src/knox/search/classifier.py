from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class QueryIntent(StrEnum):
    GREETING = "greeting"
    GRATITUDE = "gratitude"
    HELP = "help"
    OFFTOPIC = "offtopic"
    KNOWLEDGE = "knowledge"


@dataclass(frozen=True)
class ClassificationResult:
    intent: QueryIntent
    response: str | None


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
    "Hello! I can answer questions about your compiled wiki. Try asking something like 'what is flash attention?' or run `knox query --help` for more options."
)
_GRATITUDE_RESPONSE: str = "You're welcome! If you have more questions about your wiki, just ask."
_HELP_RESPONSE: str = 'I can search your compiled wiki for relevant information. Try: `knox query "your question"` or `knox query --help`.'
_OFFTOPIC_RESPONSE: str = (
    "I'm a knowledge base assistant. I can answer questions about your compiled wiki. Try asking something about the documents you've compiled."
)


_STOP: frozenset[str] = frozenset(
    {
        "the",
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


def _has_substance(query: str) -> bool:
    tokens: list[str] = re.split(r"[^a-zA-Z0-9_]+", query.strip().lower())
    content_tokens: list[str] = [t for t in tokens if len(t) > 2 and t not in _STOP]
    return len(content_tokens) >= 2


_CLASSIFIERS: list[tuple[QueryIntent, list[re.Pattern[str]], str, bool]] = [
    (QueryIntent.OFFTOPIC, _OFFTOPIC_PATTERNS, _OFFTOPIC_RESPONSE, False),
    (QueryIntent.GREETING, _GREETING_PATTERNS, _GREETING_RESPONSE, True),
    (QueryIntent.GRATITUDE, _GRATITUDE_PATTERNS, _GRATITUDE_RESPONSE, False),
    (QueryIntent.HELP, _HELP_PATTERNS, _HELP_RESPONSE, True),
]


def classify(query: str, min_length: int = 2) -> ClassificationResult:
    lower: str = query.strip().lower()

    if len(lower) < min_length:
        return ClassificationResult(QueryIntent.KNOWLEDGE, response=None)

    for intent, patterns, response, match_start in _CLASSIFIERS:
        if intent == QueryIntent.OFFTOPIC:
            for p in patterns:
                if p.search(lower):
                    return ClassificationResult(intent, response=response)
        elif _has_substance(lower):
            break
        else:
            for p in patterns:
                matched = p.match if match_start else p.search
                if matched(lower):
                    return ClassificationResult(intent, response=response)

    return ClassificationResult(QueryIntent.KNOWLEDGE, response=None)
