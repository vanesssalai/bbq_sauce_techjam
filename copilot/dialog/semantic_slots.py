from __future__ import annotations

import math
from typing import Protocol, Sequence


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


PROTOTYPES: dict[str, dict[str, list[str]]] = {
    "color": {
        "red": ["red", "burgundy", "maroon", "crimson", "wine coloured", "ruby"],
        "blue": ["blue", "navy", "cobalt", "sky blue", "teal"],
        "green": ["green", "olive", "emerald", "forest green", "sage"],
        "black": ["black", "jet black", "charcoal"],
        "white": ["white", "ivory", "cream", "off white"],
        "pink": ["pink", "blush", "rose", "fuchsia", "salmon"],
        "grey": ["grey", "gray", "slate", "heather grey"],
        "brown": ["brown", "tan", "beige", "camel", "chocolate"],
        "purple": ["purple", "violet", "lavender", "plum"],
        "yellow": ["yellow", "mustard", "gold"],
    },
    "material": {
        "cotton": ["cotton", "soft cotton", "breathable cotton jersey"],
        "leather": ["leather", "genuine leather", "full grain leather"],
        "wool": ["wool", "merino wool", "woollen knit"],
        "denim": ["denim", "jean fabric"],
        "silk": ["silk", "satin", "silky"],
        "polyester": ["polyester", "synthetic performance fabric"],
        "linen": ["linen", "lightweight linen"],
        "fleece": ["fleece", "sherpa", "cosy fleece"],
        "suede": ["suede", "brushed suede"],
        "nylon": ["nylon", "ripstop nylon"],
    },
    "style": {
        "formal": ["formal", "cocktail", "evening wear", "black tie", "dressy", "elegant"],
        "casual": ["casual", "everyday", "relaxed", "laid back", "weekend"],
        "sporty": ["sporty", "athletic", "activewear", "gym clothes"],
        "vintage": ["vintage", "retro", "old school"],
        "minimalist": ["minimalist", "simple", "clean lines", "understated"],
    },
    "use_case": {
        "running": ["running", "for runs", "jogging", "marathon training"],
        "gym": ["gym", "workout", "weightlifting", "strength training"],
        "hiking": ["hiking", "trail walking", "backpacking", "trekking"],
        "work": ["office", "work", "business", "the commute"],
        "wedding": ["wedding", "wedding guest", "a ceremony"],
        "winter": ["winter", "cold weather", "snow", "freezing temperatures"],
        "summer": ["summer", "hot weather", "the beach", "warm days"],
        "travel": ["travel", "packing light", "a trip"],
        "rain": ["rain", "wet weather", "downpours"],
    },
}


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(_dot(a, a)) or 1.0
    nb = math.sqrt(_dot(b, b)) or 1.0
    return _dot(a, b) / (na * nb)


class SemanticSlotResolver:
    def __init__(
        self,
        encoder: Encoder,
        *,
        threshold: float = 0.42,
        prototypes: dict[str, dict[str, list[str]]] = PROTOTYPES,
    ) -> None:
        self._encoder = encoder
        self._threshold = threshold
        self._prototypes = prototypes
        self._proto_vecs: dict[tuple[str, str], list[Sequence[float]]] | None = None

    def _ensure_prototypes(self) -> None:
        if self._proto_vecs is not None:
            return
        triples = [
            (slot, value, phrase)
            for slot, values in self._prototypes.items()
            for value, phrases in values.items()
            for phrase in phrases
        ]
        vecs = list(self._encoder.encode([t[2] for t in triples]))
        acc: dict[tuple[str, str], list[Sequence[float]]] = {}
        for (slot, value, _), vec in zip(triples, vecs):
            acc.setdefault((slot, value), []).append(vec)
        self._proto_vecs = acc

    def resolve(self, text: str, *, only: set[str] | None = None) -> dict[str, tuple[str, float]]:
        text = (text or "").strip()
        if not text:
            return {}
        self._ensure_prototypes()
        assert self._proto_vecs is not None
        query_vec = list(self._encoder.encode([text]))[0]

        by_slot: dict[str, list[tuple[float, str]]] = {}
        for (slot, value), proto_vecs in self._proto_vecs.items():
            if only and slot not in only:
                continue
            score = max(_cosine(query_vec, pv) for pv in proto_vecs)
            by_slot.setdefault(slot, []).append((score, value))

        out: dict[str, tuple[str, float]] = {}
        for slot, scored in by_slot.items():
            top_score, top_value = max(scored)
            if top_score >= self._threshold:
                out[slot] = (top_value, top_score)
        return out