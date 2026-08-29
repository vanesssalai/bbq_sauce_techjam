"""Turn accumulated session state into the one `Query` both retrieval channels consume.

Reads what the NLU / state-machine layer (owned elsewhere) leaves on
`SessionState`: `slots`, `negated_values`, `disclosed_phrases`, `current_track`,
`intent_p_buying`. The remaining TODO is honouring an LLM-rewritten query once
`ParsedTurn.rewritten_query` is threaded onto the session.
"""

from __future__ import annotations

import re

from copilot.contracts import Query, SessionState, Slot

_SOFT_SLOTS = ("color", "material", "brand", "size")
_LOOKING_FOR_RE = re.compile(
    r"(?:looking for|search(?:ing)? for|want|need|shopping for|find me)\s+(.*)", re.I
)


def active_slots(session: SessionState) -> dict[str, Slot]:
    """Slots currently in force: `session.slots` minus any relaxed attribute.
    (`session.slots` is already keyed by attribute, so "newest per attribute" is
    inherent.)"""
    relaxed = set(session.relaxed_attrs)
    return {attr: slot for attr, slot in session.slots.items() if attr not in relaxed}


def _category_anchor(slots: dict[str, Slot], session: SessionState) -> str:
    if "category" in slots:
        return slots["category"].value
    if session.raw_history:
        first = session.raw_history[0]
        text = first[-1] if isinstance(first, (list, tuple)) else str(first)
        match = _LOOKING_FOR_RE.search(text)
        return (match.group(1) if match else text).strip(" .!?")
    return ""


def build_query(session: SessionState) -> Query:
    slots = active_slots(session)

    anchor = _category_anchor(slots, session)
    soft_values = [slots[a].value for a in _SOFT_SLOTS if a in slots]
    # free_text = category anchor + everything the customer has literally stated
    # + soft preferences repeated x2 for weight. Negations stay structural.
    parts = [anchor, *session.disclosed_phrases, *soft_values, *soft_values]
    free_text = " ".join(p for p in parts if p).strip()
    # TODO: honour ParsedTurn.rewritten_query verbatim once it reaches the session.

    return Query(
        free_text=free_text,
        slots=slots,
        negations=dict(session.negated_values),
        intent_p_buying=session.intent_p_buying,
        track=session.current_track or "browsing",
        turn=session.turn,
    )
