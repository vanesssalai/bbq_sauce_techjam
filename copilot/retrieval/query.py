from __future__ import annotations

import re

from ..contracts import (
    HARD_FILTER_ATTRS,
    SOFT_BOOST_ATTRS,
    ParsedTurn,
    Query,
    SessionState,
    Slot,
)
from ..dialog.state_machine import active_slots, effective_confidence


_FREE_TEXT_SLOTS = ("category", "department", "style", "use_case")
_SOFT_TEXT_SLOTS = ("color", "material", "brand")

_HARD_FILTER_SOURCES = frozenset({"explicit", "clarification_answer"})
_HARD_FILTER_MIN_CONF = 0.5


_LEADIN_RE = re.compile(
    r"^\s*(?:"
    r"i'?m\s+looking\s+for|i\s+am\s+looking\s+for|looking\s+for|"
    r"i'?m\s+after|i'?m\s+shopping\s+for|i\s+need|i\s+want|i'?d\s+like|"
    r"i\s+would\s+like|show\s+me|do\s+you\s+have"
    r")\s+(?:a\s+|an\s+|some\s+|the\s+|any\s+)?",
    re.IGNORECASE,
)

_ANCHOR_TAIL_RE = re.compile(r"[,.;:!?].*$", re.DOTALL)

_WORD_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "with", "in", "on", "at",
    "i", "my", "me", "is", "are", "im", "please", "some", "that", "this", "it",
}
_SKIP_HEAD = _STOPWORDS | {
    "clothing", "apparel", "clothes", "item", "items", "products", "product",
    "something", "anything", "stuff", "gear",
}


def _category_anchor(session: SessionState) -> str:
    if not session.raw_history:
        return ""
    first = session.raw_history[0][1].strip()
    phrase = _ANCHOR_TAIL_RE.sub("", _LEADIN_RE.sub("", first, count=1)).strip(" -\"'")
    tokens = phrase.split()

    lo, hi = 0, len(tokens)
    while lo < hi and tokens[lo].lower() in _SKIP_HEAD:
        lo += 1
    while hi > lo and tokens[hi - 1].lower() in _STOPWORDS:
        hi -= 1
    tokens = tokens[lo:hi]

    if not tokens or len(tokens) > 6:
        return ""
    tokens = tokens[-4:] 
    while tokens and tokens[0].lower() in _STOPWORDS:
        tokens.pop(0)
    return " ".join(tokens)


def _dedup_join(parts: list[str]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        text = (part or "").strip()
        tokens = [t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS]
        if not tokens or all(t in seen for t in tokens):
            continue
        seen.update(tokens)
        out.append(text)
    return " ".join(out)


def build_query(session: SessionState, parsed: ParsedTurn | None = None) -> Query:
    live = active_slots(session)

    hard_slots = {
        a: s for a, s in live.items()
        if a in HARD_FILTER_ATTRS
        and s.source in _HARD_FILTER_SOURCES
        and effective_confidence(s, session.turn) >= _HARD_FILTER_MIN_CONF
    }
    soft_slots = {a: s for a, s in live.items() if a in SOFT_BOOST_ATTRS}
    negations = {a: list(v) for a, v in session.negated_values.items() if v}

    anchor = _category_anchor(session)

    parts: list[str] = []
    if parsed is not None and parsed.rewritten_query:
        parts.append(parsed.rewritten_query)
    parts.append(anchor)
    parts.extend(live[a].value for a in _FREE_TEXT_SLOTS if a in live)
    parts.extend(session.disclosed_phrases)
    parts.extend(soft_slots[a].value for a in _SOFT_TEXT_SLOTS if a in soft_slots)
    if parsed is not None:
        parts.extend(parsed.soft_tags)

    free_text = _dedup_join(parts)
    if not free_text and session.raw_history:
        free_text = session.raw_history[-1][1].strip()

    return Query(
        hard_slots=hard_slots,
        soft_slots=soft_slots,
        negations=negations,
        free_text=free_text,
        category_anchor=anchor,
        intent_p_buying=session.intent_p_buying,
    )


if __name__ == "__main__":
    demo = SessionState(session_id="demo", turn=2)
    demo.raw_history = [
        ("user", "I'm looking for Earrings Hoop, but I'm still exploring."),
        ("user", "For that, what matters is: sterling silver; hypoallergenic posts."),
    ]
    demo.slots = {
        "category": Slot("earrings", 0.8, 1, "explicit"),
        "material": Slot("silver", 0.75, 2, "clarification_answer"),
        "price_max": Slot("50.00", 0.85, 1, "explicit"),
    }
    demo.disclosed_phrases = ["sterling silver", "hypoallergenic posts"]
    demo.negated_values = {"color": ["gold"]}

    q = build_query(demo)
    print("category_anchor:", repr(q.category_anchor))
    print("hard_slots     :", {k: v.value for k, v in q.hard_slots.items()})
    print("soft_slots     :", {k: v.value for k, v in q.soft_slots.items()})
    print("negations      :", q.negations)
    print("free_text      :", repr(q.free_text))
