from __future__ import annotations

import os
from dataclasses import replace

from ..contracts import (
    ASK_ATTRIBUTE_TO_SLOTS,
    SLOT_TO_ASK_ATTRIBUTE,
    ParsedTurn,
    SessionState,
    Slot,
)

_DECAY_RATE = 0.85
_STALE_THRESHOLD = 0.3
_TRACK_SWITCH_CONFIDENCE = 0.6
_OVERRIDE_CONFIDENCE = 0.95
_TURN_BUDGET_CRITICAL = 2

_PINNED_CONFIDENCE = 0.6 


try:
    _MAX_CLARIFICATIONS = int(os.environ.get("COPILOT_MAX_CLARIFY", "6"))
except ValueError:
    _MAX_CLARIFICATIONS = 6
_SECOND_QUESTION_MIN_REMAINING = 2

_KEEP_ON_HARD_RESET = {"category", "department"} 

_INTENT_EMA_ALPHA = 0.35
_MAX_SESSION_PHRASES = 16 


def apply_turn(session: SessionState, parsed: ParsedTurn) -> SessionState:
    pending_ask = session.pending_ask_attribute
    pending_relax = session.pending_relaxation

    session.turn = parsed.turn
    session.raw_history.append(("user", parsed.raw_text))
    session.pending_ask_attribute = None
    session.pending_relaxation = None

    if parsed.is_no_preference:
        _record_no_preference(session, parsed, pending_ask)

    if parsed.is_hard_reset:
        _apply_hard_reset(session, parsed)

    _update_track(session, parsed)
    _update_intent_confidence(session, parsed)
    _merge_slots(session, parsed)
    _merge_negations(session, parsed)
    _reconcile_price(session)
    _resolve_pending_relaxation(session, parsed, pending_relax)
    _accumulate_phrases(session, parsed)

    return session


def _update_intent_confidence(session: SessionState, parsed: ParsedTurn) -> None:
    if parsed.intent_tier == "none" and abs(parsed.intent_p_buying - 0.5) < 0.02:
        return
    a = _INTENT_EMA_ALPHA
    session.intent_p_buying = (1.0 - a) * session.intent_p_buying + a * parsed.intent_p_buying


def _resolve_pending_relaxation(
    session: SessionState, parsed: ParsedTurn, pending: tuple[str, str | None] | None
) -> None:
    if not pending:
        return
    attr, new_value = pending
    if parsed.is_affirmation and not parsed.is_rejection:
        _apply_relaxation(session, attr, new_value)
    elif parsed.is_rejection:
        slot = session.slots.get(attr)
        if slot and attr not in parsed.slots:   # customer held firm and didn't restate
            session.slots[attr] = replace(slot, confidence=1.0, turn_set=session.turn)


def _apply_relaxation(session: SessionState, attr: str, new_value: str | None) -> None:
    if attr in ("price_min", "price_max") and new_value is not None:
        session.slots[attr] = Slot(new_value, 0.7, session.turn, "inferred")
    else:
        session.slots.pop(attr, None)
    if attr not in session.relaxed_attrs:
        session.relaxed_attrs.append(attr)


def record_shown(session: SessionState, parent_asins) -> None:
    session.shown_asins.update(parent_asins)


def _record_no_preference(session: SessionState, parsed: ParsedTurn, pending_ask: str | None) -> None:
    attr = parsed.no_preference_attribute or pending_ask
    if attr:
        session.no_preference.add(attr)


def _apply_hard_reset(session: SessionState, parsed: ParsedTurn) -> None:
    session.disclosed_phrases.clear()
    restated = set(parsed.slots)
    for attr in list(session.slots):
        if attr in _KEEP_ON_HARD_RESET or attr in restated:
            continue
        del session.slots[attr]
    session.negated_values.clear()


def _update_track(session: SessionState, parsed: ParsedTurn) -> None:
    if session.current_track is None or parsed.intent_confidence >= _TRACK_SWITCH_CONFIDENCE:
        session.current_track = parsed.intent


def _merge_slots(session: SessionState, parsed: ParsedTurn) -> None:
    for attr, new_slot in parsed.slots.items():
        if parsed.is_override and attr in parsed.overridden_attrs:
            new_slot = replace(new_slot, confidence=max(new_slot.confidence, _OVERRIDE_CONFIDENCE))
        session.slots[attr] = new_slot

        negated = session.negated_values.get(attr)
        if negated and new_slot.value in negated:
            negated.remove(new_slot.value)

        ask_attr = SLOT_TO_ASK_ATTRIBUTE.get(attr)
        if ask_attr:
            session.no_preference.discard(ask_attr)


def _merge_negations(session: SessionState, parsed: ParsedTurn) -> None:
    for attr, values in parsed.negated_values.items():
        existing = session.negated_values.setdefault(attr, [])
        for value in values:
            if value not in existing:
                existing.append(value)


def _reconcile_price(session: SessionState) -> None:
    lo = session.slots.get("price_min")
    hi = session.slots.get("price_max")
    if not (lo and hi):
        return
    try:
        if float(lo.value) <= float(hi.value):
            return
    except ValueError:
        return
    if lo.turn_set >= hi.turn_set:
        del session.slots["price_max"]
    else:
        del session.slots["price_min"]


def _accumulate_phrases(session: SessionState, parsed: ParsedTurn) -> None:
    seen = {p.lower() for p in session.disclosed_phrases}
    for phrase in parsed.disclosed_phrases:
        text = phrase.strip()
        if text and text.lower() not in seen:
            session.disclosed_phrases.append(text)
            seen.add(text.lower())
    if len(session.disclosed_phrases) > _MAX_SESSION_PHRASES:
        del session.disclosed_phrases[:-_MAX_SESSION_PHRASES]


def effective_confidence(slot: Slot, current_turn: int) -> float:
    turns_elapsed = max(0, current_turn - slot.turn_set)
    return slot.confidence * (_DECAY_RATE ** turns_elapsed)


def active_slots(session: SessionState) -> dict[str, Slot]:
    return {
        attr: slot
        for attr, slot in session.slots.items()
        if effective_confidence(slot, session.turn) >= _STALE_THRESHOLD
    }


def turns_remaining(session: SessionState, max_turns: int = 10) -> int:
    return max(0, max_turns - session.turn)


def is_turn_budget_critical(session: SessionState, max_turns: int = 10) -> bool:
    return turns_remaining(session, max_turns) <= _TURN_BUDGET_CRITICAL


def note_clarification(session: SessionState) -> None:
    session.clarify_count += 1


def should_ask(
    session: SessionState,
    attribute: str | None = None,
    *,
    candidates_ambiguous: bool = True,
    max_turns: int = 10,
) -> bool:
    if is_turn_budget_critical(session, max_turns):
        return False
    if session.clarify_count >= _MAX_CLARIFICATIONS:
        return False
    if session.clarify_count >= 1 and turns_remaining(session, max_turns) < _SECOND_QUESTION_MIN_REMAINING:
        return False
    if not candidates_ambiguous:
        return False
    if attribute is not None:
        if attribute in session.no_preference:
            return False
        for slot_key in ASK_ATTRIBUTE_TO_SLOTS.get(attribute, [attribute]):
            slot = session.slots.get(slot_key)
            if slot and effective_confidence(slot, session.turn) >= _PINNED_CONFIDENCE:
                return False
    return True
