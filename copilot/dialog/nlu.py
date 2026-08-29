from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from ..contracts import ASK_ATTRIBUTE_TO_SLOTS, ParsedTurn, Slot
from .fuzzy import best_fuzzy_match

if TYPE_CHECKING:
    from .intent import EmbeddingIntentScorer
    from .semantic_slots import SemanticSlotResolver
from ..vocab import (
    AFFIRMATION_PHRASES,
    AFFIRMATION_STARTS,
    BRAND_WORDS,
    CATEGORY_WORDS,
    COLOR_WORDS,
    COMPARATIVE_SLOT_SHIFT,
    DEPARTMENT_SYNONYMS,
    DISSATISFACTION_CUES,
    HARD_RESET_CUES,
    INTENT_CUES,
    LETTER_SIZE_ORDER,
    MATERIAL_WORDS,
    NEGATION_CUES,
    NO_PREFERENCE_CUES,
    OVERRIDE_CUES,
    REJECTION_PHRASES,
    REJECTION_STARTS,
    SIZE_DOWN_CUES,
    SIZE_LETTER_RE,
    SIZE_NUMERIC_RE,
    SIZE_UP_CUES,
    SIZE_WORD_LOOSE_RE,
    SIZE_WORD_MAP,
    SOFT_TAG_WORDS,
)

_PRICE_RANGE_RE = re.compile(
    r"\$?(\d+(?:\.\d+)?)\s*(?:-|to|and)\s*\$?(\d+(?:\.\d+)?)", re.IGNORECASE
)
_PRICE_UNDER_RE = re.compile(
    r"\b(?:under|below|less than|no more than|up to|cheaper than)\s*\$?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_PRICE_OVER_RE = re.compile(
    r"\b(?:over|above|more than|at least)\s*\$?(\d+(?:\.\d+)?)", re.IGNORECASE
)
_PRICE_AROUND_RE = re.compile(
    r"\b(?:around|about|roughly|budget of)\s*\$?(\d+(?:\.\d+)?)", re.IGNORECASE
)
_PRICE_COMPLAINT_WORDS = ["expensive", "pricey", "pricy", "cost", "costly", "overpriced"]

_NEGATABLE_VOCAB = {"color": COLOR_WORDS, "material": MATERIAL_WORDS}
_NEGATION_WINDOW = 20 


_PHRASE_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "am",
    "for", "of", "and", "or", "to", "in", "into", "on", "at", "with", "without",
    "this", "that", "these", "those", "there", "here", "it", "its", "my", "me",
    "i", "you", "your", "we", "us", "our", "im", "i'm", "ive", "i've", "id",
    "i'd", "what", "which", "who", "do", "does", "did", "have", "has", "had",
    "looking", "look", "want", "wanted", "need", "needed", "needs", "like",
    "would", "could", "should", "still", "but", "just", "some", "something",
    "anything", "really", "actually", "instead", "also", "well", "please",
    "thanks", "thank", "make", "made", "get", "got", "show", "find", "help",
    "prefer", "prefers", "yes", "yeah", "yep", "no", "nope", "ok", "okay",
    "hmm", "so", "then", "now", "maybe", "kind", "sort", "bit",
}

_PHRASE_BOILERPLATE = (
    "still exploring", "just exploring", "just browsing", "just looking",
    "ask me about", "one specific attribute", "not quite right",
    "options are not", "use your judgment", "use your judgement",
    "please use your", "your best judgment", "leave it to you",
    "don't have a preference", "dont have a preference",
    "do not have a preference", "don't have an additional preference",
    "dont have an additional preference", "no additional preference",
    "ignore my earlier", "ignore my previous", "ignore what i said",
    "forget what i said", "no preference",
)


_PHRASE_LEADIN_RE = re.compile(
    r"^(?:for that[,:]?\s*)?(?:"
    r"a key requirement is|key requirement is|what matters is|what i need is|"
    r"what i'm looking for is|what im looking for is|i'm looking for|"
    r"im looking for|i am looking for|looking for|i need|i want|i'd like|"
    r"i would like|i really like|i like|my requirement is|the requirement is|"
    r"i'm after|im after|i'm shopping for|im shopping for"
    r")\b[:\s]*",
    re.IGNORECASE,
)
_PHRASE_CHUNK_SPLIT_RE = re.compile(r"[;,]| and ")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PHRASE_WORD_RE = re.compile(r"[a-z0-9']+")
_PHRASE_NEG_PREFIXES = (
    "no ", "not ", "non ", "don't ", "dont ", "without ", "except ",
    "other than ", "excluding ", "aside from ",
)
_MAX_DISCLOSED_PHRASES = 12

_NO_PREF_ATTR_RE = re.compile(
    r"preference\s+(?:for|on|about|regarding|as to)\s+(?:the\s+|a\s+|an\s+)?([a-z_]+)"
)
_ALLOWED_ASK_ATTRS = frozenset(ASK_ATTRIBUTE_TO_SLOTS)


def _scan_with_positions(text_lower: str, vocab: list[str]) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for word in vocab:
        m = re.search(rf"\b{re.escape(word)}\b", text_lower)
        if m:
            matches.append((m.start(), word))
    matches.sort(key=lambda pair: pair[0])
    return matches


def _is_negated(text_lower: str, pos: int) -> bool:
    window = text_lower[max(0, pos - _NEGATION_WINDOW):pos]
    return any(cue in window for cue in NEGATION_CUES)


def _extract_negated_values(text_lower: str) -> dict[str, list[str]]:
    negated: dict[str, list[str]] = {}
    for attr, vocab in _NEGATABLE_VOCAB.items():
        hits = [word for pos, word in _scan_with_positions(text_lower, vocab) if _is_negated(text_lower, pos)]
        if hits:
            negated[attr] = hits
    return negated


def _extract_colors(text_lower: str, negated: set[str]) -> list[str]:
    return [word for _, word in _scan_with_positions(text_lower, COLOR_WORDS) if word not in negated]


def _extract_materials(text_lower: str, negated: set[str]) -> list[str]:
    return [word for _, word in _scan_with_positions(text_lower, MATERIAL_WORDS) if word not in negated]


def _extract_brand(text_lower: str) -> str | None:
    matches = _scan_with_positions(text_lower, BRAND_WORDS)
    return matches[0][1] if matches else None


_FUZZY_TOKEN_RE = re.compile(r"[a-z0-9']{5,}")
_BRAND_FUZZY_THRESHOLD = 0.58


_FUZZY_BRAND_STOPWORDS = (
    set(COLOR_WORDS) | set(MATERIAL_WORDS) | set(CATEGORY_WORDS) | set(SOFT_TAG_WORDS)
    | {
        "please", "around", "really", "looking", "something", "anything",
        "instead", "actually", "prefer", "budget", "cheaper", "smaller",
        "bigger", "larger", "warmer", "lighter", "dressier", "little", "would",
        "could", "should", "maybe", "thanks", "thank", "these", "those",
        "under", "about", "which", "where", "whatever", "options", "another",
    }
)


def _fuzzy_brand(text_lower: str) -> str | None:
    tokens = [t for t in _FUZZY_TOKEN_RE.findall(text_lower) if t not in _FUZZY_BRAND_STOPWORDS]
    if not tokens:
        return None
    candidates = list(tokens)
    candidates += [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    candidates += [f"{a}{b}" for a, b in zip(tokens, tokens[1:])]

    best: str | None = None
    best_score = _BRAND_FUZZY_THRESHOLD
    for cand in candidates:
        match = best_fuzzy_match(cand, BRAND_WORDS, threshold=best_score, min_len=5)
        if match:
            best, best_score = match
    return best


def _extract_sizes(text: str) -> list[str]:
    text_lower = text.lower()
    letters = [s.upper() for s in SIZE_LETTER_RE.findall(text)]
    numerics = SIZE_NUMERIC_RE.findall(text)
    words = [SIZE_WORD_MAP[w.lower()] for w in SIZE_WORD_LOOSE_RE.findall(text_lower)]

    seen: set[str] = set()
    result: list[str] = []
    for s in [*letters, *numerics, *words]:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _extract_department(text_lower: str) -> str | None:
    for word, canonical in DEPARTMENT_SYNONYMS.items():
        if re.search(rf"\b{re.escape(word)}\b", text_lower):
            return canonical
    return None


def _extract_category(text_lower: str) -> str | None:
    matches: list[tuple[int, str]] = []
    for word in CATEGORY_WORDS:
        m = re.search(rf"\b{re.escape(word)}(?:es|s)?\b", text_lower)
        if m:
            matches.append((m.start(), word))
    matches.sort(key=lambda pair: pair[0])
    return matches[0][1] if matches else None


def _extract_price_range(text: str) -> tuple[float | None, float | None]:
    range_match = _PRICE_RANGE_RE.search(text)
    if range_match:
        a, b = float(range_match.group(1)), float(range_match.group(2))
        return min(a, b), max(a, b)

    under_match = _PRICE_UNDER_RE.search(text)
    if under_match:
        return None, float(under_match.group(1))

    over_match = _PRICE_OVER_RE.search(text)
    if over_match:
        return float(over_match.group(1)), None

    around_match = _PRICE_AROUND_RE.search(text)
    if around_match:
        return None, float(around_match.group(1))

    return None, None


def _extract_soft_tags(text_lower: str) -> list[str]:
    return [word for _, word in _scan_with_positions(text_lower, SOFT_TAG_WORDS)]


def _detect_override(text_lower: str) -> bool:
    return any(cue in text_lower for cue in OVERRIDE_CUES)


def _classify_dissatisfaction_reason(text_lower: str) -> str:
    if any(word in text_lower for word in _PRICE_COMPLAINT_WORDS):
        return "budget"
    if _scan_with_positions(text_lower, COLOR_WORDS):
        return "color"
    if _scan_with_positions(text_lower, MATERIAL_WORDS):
        return "material"
    if _extract_category(text_lower):
        return "category"
    return "other"


def _detect_dissatisfaction(text_lower: str) -> tuple[bool, str | None]:
    if not any(cue in text_lower for cue in DISSATISFACTION_CUES):
        return False, None
    return True, _classify_dissatisfaction_reason(text_lower)


_TIER_TARGET_P = {
    "transactional": 0.90,
    "navigational": 0.70,
    "commercial": 0.30,
    "informational": 0.15,
}


def _score_track_cues(
    text_lower: str, prior_track: Literal["buying", "browsing"] | None
) -> tuple[Literal["buying", "browsing"], float, str]:
    hits = {tier: sum(1 for cue in cues if cue in text_lower) for tier, cues in INTENT_CUES.items()}
    total = sum(hits.values())
    if total == 0:
        return (prior_track or "buying"), 0.5, "none"

    p = sum(_TIER_TARGET_P[tier] * n for tier, n in hits.items()) / total
    counts = sorted(hits.values())
    margin = counts[-1] - counts[-2]  # how lopsided the tier evidence is
    p += 0.04 * min(margin, 3) * (1.0 if p >= 0.5 else -1.0)
    p = min(0.95, max(0.05, p))

    dominant = max(hits, key=hits.get)
    label: Literal["buying", "browsing"] = "buying" if p >= 0.5 else "browsing"
    return label, min(0.95, 0.5 + abs(p - 0.5)), dominant


def _score_track(
    text: str,
    text_lower: str,
    prior_track: Literal["buying", "browsing"] | None,
    intent_scorer: "EmbeddingIntentScorer | None" = None,
) -> tuple[Literal["buying", "browsing"], float, float, str]:
    cue_label, cue_conf, cue_tier = _score_track_cues(text_lower, prior_track)
    cue_p = cue_conf if cue_label == "buying" else 1.0 - cue_conf

    if intent_scorer is None:
        return cue_label, cue_conf, cue_p, cue_tier

    try:
        _, emb_p = intent_scorer.score(text)
    except Exception:
        return cue_label, cue_conf, cue_p, cue_tier

    cue_strength = abs(cue_p - 0.5) * 2.0
    if cue_strength == 0.0 and abs(emb_p - 0.5) < 0.05:
        return (prior_track or "buying"), 0.5, 0.5, cue_tier

    w_cue = 0.5 + 0.4 * cue_strength  # 0.5 .. 0.9
    p_buying = min(0.99, max(0.01, w_cue * cue_p + (1.0 - w_cue) * emb_p))
    label: Literal["buying", "browsing"] = "buying" if p_buying >= 0.5 else "browsing"
    confidence = min(0.95, 0.5 + abs(p_buying - 0.5))
    return label, confidence, p_buying, cue_tier


_ATTR_EXTRACTORS = {
    "color": lambda text_lower, negated: (_extract_colors(text_lower, negated.get("color", set())) or [None])[0],
    "material": lambda text_lower, negated: (_extract_materials(text_lower, negated.get("material", set())) or [None])[0],
    "size": lambda text_lower, _: (_extract_sizes(text_lower) or [None])[0],
    "department": lambda text_lower, _: _extract_department(text_lower),
    "category": lambda text_lower, _: _extract_category(text_lower),
    "brand": lambda text_lower, _: _extract_brand(text_lower),
}


def _bind_pending_answer(
    text: str,
    text_lower: str,
    pending_ask_attribute: str | None,
    turn: int,
    general_slots: dict[str, Slot],
) -> dict[str, Slot]:
    if not pending_ask_attribute:
        return {}

    target_slots = ASK_ATTRIBUTE_TO_SLOTS.get(pending_ask_attribute, [])
    if not target_slots:
        return {}

    if pending_ask_attribute == "budget":
        price_min, price_max = _extract_price_range(text)
        bound: dict[str, Slot] = {}
        if price_min is not None:
            bound["price_min"] = Slot(f"{price_min:.2f}", 0.9, turn, "clarification_answer")
        if price_max is not None:
            bound["price_max"] = Slot(f"{price_max:.2f}", 0.9, turn, "clarification_answer")
        return bound

    bound = {}
    for slot_key in target_slots:
        if slot_key in general_slots:
            found = general_slots[slot_key]
            bound[slot_key] = Slot(found.value, max(found.confidence, 0.9), turn, "clarification_answer")

    if not bound and len(text.split()) <= 4:
        bound[target_slots[0]] = Slot(text.strip(" .!?"), 0.6, turn, "clarification_answer")

    return bound


def _extract_disclosed_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
        stripped = _PHRASE_LEADIN_RE.sub("", sentence.strip()).strip(" .!?:-\"'")
        if not stripped:
            continue
        for chunk in _PHRASE_CHUNK_SPLIT_RE.split(stripped):
            chunk = chunk.strip(" .!?:-\"'")
            low = chunk.lower()
            if len(chunk) < 3:
                continue
            if any(marker in low for marker in _PHRASE_BOILERPLATE):
                continue
            if any(low.startswith(prefix) for prefix in _PHRASE_NEG_PREFIXES):
                continue
            if not any(w not in _PHRASE_STOPWORDS for w in _PHRASE_WORD_RE.findall(low)):
                continue
            if low in seen:
                continue
            seen.add(low)
            phrases.append(chunk)
            if len(phrases) >= _MAX_DISCLOSED_PHRASES:
                return phrases
    return phrases


def _detect_hard_reset(text_lower: str) -> bool:
    return any(cue in text_lower for cue in HARD_RESET_CUES)


def _detect_no_preference(text_lower: str) -> tuple[bool, str | None]:
    if not any(cue in text_lower for cue in NO_PREFERENCE_CUES):
        return False, None
    match = _NO_PREF_ATTR_RE.search(text_lower)
    attr = match.group(1) if match else None
    if attr not in _ALLOWED_ASK_ATTRS:
        attr = None
    return True, attr

_LLM_HINT_SLOT_KEYS = {
    "category", "department", "color", "material", "size", "brand",
    "price_min", "price_max", "style", "use_case",
}


def _merge_llm_hint(rule_slots: dict[str, Slot], llm_hint: dict | None, turn: int) -> dict[str, Slot]:
    if not llm_hint:
        return rule_slots
    hint_slots = llm_hint.get("slots")
    if not isinstance(hint_slots, dict):
        return rule_slots
    merged = dict(rule_slots)
    for attr, value in hint_slots.items():
        if attr not in _LLM_HINT_SLOT_KEYS or attr in merged or value in (None, ""):
            continue
        merged[attr] = Slot(str(value).strip(), 0.7, turn, "llm")
    return merged


def _hint_query(llm_hint: dict | None) -> str | None:
    if not llm_hint:
        return None
    query = llm_hint.get("query")
    query = str(query).strip() if query else ""
    return query or None


_CHEAPER_CUES = (
    "cheaper", "less expensive", "more affordable", "lower price", "lower budget",
    "budget friendly", "not so expensive", "too expensive", "too pricey",
    "bring the price down",
)
_PRICIER_CUES = (
    "pricier", "more expensive", "higher end", "premium", "spend more",
    "willing to pay more", "bump my budget", "go higher on price", "nicer ones",
)


def _prior_price(prior_slots: dict[str, str] | None, key: str) -> float | None:
    if not prior_slots:
        return None
    try:
        raw = prior_slots.get(key)
        return float(raw) if raw not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _apply_comparative_price(
    slots: dict[str, Slot], text_lower: str, prior_slots: dict[str, str] | None, turn: int
) -> dict[str, Slot]:
    wants_cheaper = any(c in text_lower for c in _CHEAPER_CUES)
    wants_pricier = any(c in text_lower for c in _PRICIER_CUES)
    if wants_cheaper == wants_pricier:  # neither, or contradictory
        return slots
    merged = dict(slots)
    if wants_cheaper and "price_max" not in merged:
        base = _prior_price(prior_slots, "price_max") or _prior_price(prior_slots, "price_min")
        if base:
            merged["price_max"] = Slot(f"{base * 0.75:.2f}", 0.6, turn, "inferred")
    if wants_pricier and "price_min" not in merged:
        base = _prior_price(prior_slots, "price_min") or _prior_price(prior_slots, "price_max")
        if base:
            merged["price_min"] = Slot(f"{base * 1.25:.2f}", 0.6, turn, "inferred")
    return merged


def _shift_size(prior: str, step: int) -> str | None:
    up = prior.upper()
    if up in LETTER_SIZE_ORDER:
        i = LETTER_SIZE_ORDER.index(up) + step
        return LETTER_SIZE_ORDER[i] if 0 <= i < len(LETTER_SIZE_ORDER) else None
    try:
        return f"{float(prior) + step:g}"
    except ValueError:
        return None


def _apply_comparatives(
    slots: dict[str, Slot], text_lower: str, prior_slots: dict[str, str] | None, turn: int
) -> dict[str, Slot]:
    merged = dict(slots)
    for phrase, (slot_key, value) in COMPARATIVE_SLOT_SHIFT.items():
        if phrase in text_lower and slot_key not in merged:
            merged[slot_key] = Slot(value, 0.6, turn, "inferred")

    if "size" not in merged and prior_slots and prior_slots.get("size"):
        want_up = any(c in text_lower for c in SIZE_UP_CUES)
        want_down = any(c in text_lower for c in SIZE_DOWN_CUES)
        if want_up != want_down:
            new = _shift_size(prior_slots["size"], 1 if want_up else -1)
            if new:
                merged["size"] = Slot(new, 0.6, turn, "inferred")
    return merged


def _detect_affirmation(text_lower: str) -> bool:
    if any(p in text_lower for p in AFFIRMATION_PHRASES):
        return True
    words = text_lower.strip(" .!?").split()
    return bool(words) and len(words) <= 6 and words[0] in AFFIRMATION_STARTS


def _detect_rejection(text_lower: str) -> bool:
    if any(p in text_lower for p in REJECTION_PHRASES):
        return True
    words = text_lower.strip(" .!?").split()
    return bool(words) and len(words) <= 6 and words[0] in REJECTION_STARTS


def _apply_semantic_slots(
    slots: dict[str, Slot], text: str, resolver: "SemanticSlotResolver | None", turn: int
) -> dict[str, Slot]:
    if resolver is None:
        return slots
    try:
        resolved = resolver.resolve(text)
    except Exception:
        return slots
    merged = dict(slots)
    for slot_key, (value, score) in resolved.items():
        if slot_key in merged:
            continue
        merged[slot_key] = Slot(value, min(0.7, 0.35 + float(score)), turn, "inferred")
    return merged


def extract_slots_and_intent(
    user_message: str,
    turn: int,
    pending_ask_attribute: str | None = None,
    prior_track: Literal["buying", "browsing"] | None = None,
    *,
    intent_scorer: "EmbeddingIntentScorer | None" = None,
    llm_hint: dict | None = None,
    semantic_resolver: "SemanticSlotResolver | None" = None,
    prior_slots: dict[str, str] | None = None,
) -> ParsedTurn:
    text = user_message.strip()
    text_lower = text.lower()

    is_affirmation = _detect_affirmation(text_lower)
    is_rejection = _detect_rejection(text_lower)

    negated_values = _extract_negated_values(text_lower)
    negated_words = {attr: set(words) for attr, words in negated_values.items()}

    general_slots: dict[str, Slot] = {}
    for attr, extractor in _ATTR_EXTRACTORS.items():
        value = extractor(text_lower, negated_words)
        if value:
            confidence = 0.75 if attr == "brand" else 0.8
            general_slots[attr] = Slot(value, confidence, turn, "explicit")

    if "brand" not in general_slots and not (is_affirmation or is_rejection):
        fuzzy = _fuzzy_brand(text_lower)
        if fuzzy:
            general_slots["brand"] = Slot(fuzzy, 0.55, turn, "inferred")

    price_min, price_max = _extract_price_range(text)
    if price_min is not None:
        general_slots["price_min"] = Slot(f"{price_min:.2f}", 0.85, turn, "explicit")
    if price_max is not None:
        general_slots["price_max"] = Slot(f"{price_max:.2f}", 0.85, turn, "explicit")

    pending_bound = _bind_pending_answer(text, text_lower, pending_ask_attribute, turn, general_slots)
    slots = {**general_slots, **pending_bound}
    slots = _merge_llm_hint(slots, llm_hint, turn)
    slots = _apply_comparative_price(slots, text_lower, prior_slots, turn)
    slots = _apply_comparatives(slots, text_lower, prior_slots, turn)
    slots = _apply_semantic_slots(slots, text, semantic_resolver, turn)

    is_hard_reset = _detect_hard_reset(text_lower)
    is_override = _detect_override(text_lower) or is_hard_reset
    overridden_attrs = list(slots.keys()) if is_override else []

    intent, intent_confidence, intent_p_buying, intent_tier = _score_track(
        text, text_lower, prior_track, intent_scorer
    )
    hint_p_buying = (llm_hint or {}).get("intent_p_buying")
    if isinstance(hint_p_buying, (int, float)) and 0.0 <= float(hint_p_buying) <= 1.0:
        intent_p_buying = 0.5 * intent_p_buying + 0.5 * float(hint_p_buying)
        intent = "buying" if intent_p_buying >= 0.5 else "browsing"
        intent_confidence = max(intent_confidence, min(0.95, 0.5 + abs(intent_p_buying - 0.5)))

    soft_tags = _extract_soft_tags(text_lower)
    is_dissatisfied, dissatisfaction_attribute = _detect_dissatisfaction(text_lower)
    disclosed_phrases = _extract_disclosed_phrases(text)
    is_no_preference, no_preference_attribute = _detect_no_preference(text_lower)

    return ParsedTurn(
        raw_text=user_message,
        turn=turn,
        intent=intent,
        intent_confidence=intent_confidence,
        is_override=is_override,
        overridden_attrs=overridden_attrs,
        slots=slots,
        soft_tags=soft_tags,
        negated_values=negated_values,
        answered_ask_attribute=pending_ask_attribute if pending_bound else None,
        is_dissatisfied=is_dissatisfied,
        dissatisfaction_attribute=dissatisfaction_attribute,
        disclosed_phrases=disclosed_phrases,
        is_no_preference=is_no_preference,
        no_preference_attribute=no_preference_attribute,
        is_hard_reset=is_hard_reset,
        intent_p_buying=intent_p_buying,
        rewritten_query=_hint_query(llm_hint),
        intent_tier=intent_tier,
        is_affirmation=is_affirmation,
        is_rejection=is_rejection,
    )


if __name__ == "__main__":
    import sys

    message = " ".join(sys.argv[1:]) or "I need a red cotton Nike dress under $40, not leather"
    parsed = extract_slots_and_intent(message, turn=1)
    print(f"raw_text: {parsed.raw_text!r}")
    print(f"intent: {parsed.intent} (confidence {parsed.intent_confidence:.2f})")
    print(f"is_override: {parsed.is_override}")
    print("slots:")
    for attr, slot in parsed.slots.items():
        print(f"  {attr:12s} = {slot.value!r:20s} (confidence {slot.confidence:.2f}, source={slot.source})")
    print(f"negated_values: {parsed.negated_values}")
    print(f"soft_tags: {parsed.soft_tags}")
    print(f"is_dissatisfied: {parsed.is_dissatisfied} (attribute={parsed.dissatisfaction_attribute})")
    print(f"disclosed_phrases: {parsed.disclosed_phrases}")
    print(f"is_no_preference: {parsed.is_no_preference} (attribute={parsed.no_preference_attribute})")
    print(f"is_hard_reset: {parsed.is_hard_reset}")
    print(f"intent_p_buying: {parsed.intent_p_buying:.2f} (tier={parsed.intent_tier})")
    print(f"rewritten_query: {parsed.rewritten_query!r}")
    print(f"is_affirmation: {parsed.is_affirmation}  is_rejection: {parsed.is_rejection}")