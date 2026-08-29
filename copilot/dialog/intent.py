from __future__ import annotations

import math
from typing import Literal, Sequence

from ..models import Encoder

Track = Literal["buying", "browsing"]

# cold-start prototype anchors -->  uses cosind similarity
BUYING_ANCHORS: list[str] = [
    "I need a specific item and it has to meet these requirements.",
    "Looking for a black leather crossbody bag, medium size.",
    "I want to buy running shoes with arch support, ready to order.",
    "Looking for a waterproof winter jacket, men's large, under $80.",
    "I know what I want: a hypoallergenic gold hoop earring set.",
]
BROWSING_ANCHORS: list[str] = [
    "I'm looking for a gift but I'm still exploring options.",
    "Just browsing to see what's available in jackets.",
    "Not sure exactly what I want yet, show me some ideas.",
    "I'm open to suggestions, what do you have for summer?",
    "Still figuring out what style I like, what would you recommend?",
]

_SIM_GAIN = 6.0


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a)) or 1.0


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    vectors = list(vectors)
    if not vectors:
        raise ValueError("centroid of an empty vector set")
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += x
    mean = [x / len(vectors) for x in acc]
    n = _norm(mean)
    return [x / n for x in mean]


class EmbeddingIntentScorer:
    def __init__(
        self,
        encoder: Encoder | None = None,
        *,
        buy_vectors: Sequence[Sequence[float]] | None = None,
        browse_vectors: Sequence[Sequence[float]] | None = None,
        buy_anchors: Sequence[str] = BUYING_ANCHORS,
        browse_anchors: Sequence[str] = BROWSING_ANCHORS,
        gain: float = _SIM_GAIN,
    ) -> None:
        have_vecs = buy_vectors is not None and browse_vectors is not None
        if not have_vecs and encoder is None:
            raise ValueError(
                "provide an encoder, or both buy_vectors and browse_vectors"
            )
        self._encoder = encoder
        self._gain = gain
        self._buy_anchors = list(buy_anchors)
        self._browse_anchors = list(browse_anchors)
        self._buy_vecs: list[Sequence[float]] | None = (
            [list(v) for v in buy_vectors] if have_vecs else None
        )
        self._browse_vecs: list[Sequence[float]] | None = (
            [list(v) for v in browse_vectors] if have_vecs else None
        )

    def _ensure_anchors(self) -> None:
        if self._buy_vecs is not None and self._browse_vecs is not None:
            return
        if self._encoder is None:  # pragma: no cover - guarded in __init__
            raise RuntimeError("no encoder available to embed the text anchors")
        vecs = list(self._encoder.encode([*self._buy_anchors, *self._browse_anchors]))
        n = len(self._buy_anchors)
        self._buy_vecs, self._browse_vecs = vecs[:n], vecs[n:]

    def anchor_vectors(self) -> tuple[list[Sequence[float]], list[Sequence[float]]]:
        self._ensure_anchors()
        assert self._buy_vecs is not None and self._browse_vecs is not None
        return list(self._buy_vecs), list(self._browse_vecs)

    def score(self, message: str) -> tuple[Track, float]:
        text = (message or "").strip()
        if not text:
            return "browsing", 0.5
        if self._encoder is None:
            raise RuntimeError("score(text) needs an encoder; use score_vector()")
        self._ensure_anchors()
        vec = list(self._encoder.encode([text]))[0]
        return self.score_vector(vec)

    def score_vector(self, vector: Sequence[float]) -> tuple[Track, float]:
        self._ensure_anchors()
        assert self._buy_vecs is not None and self._browse_vecs is not None
        buy_sim = max(_cosine(vector, a) for a in self._buy_vecs)
        browse_sim = max(_cosine(vector, a) for a in self._browse_vecs)
        p_buying = _sigmoid(self._gain * (buy_sim - browse_sim))
        p_buying = min(0.99, max(0.01, p_buying))
        return ("buying" if p_buying >= 0.5 else "browsing"), p_buying


def load_anchors(path: str = "data/intent_anchors.json") -> tuple[list, list] | None:
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return list(data["buy_vectors"]), list(data["browse_vectors"])
    except (ValueError, KeyError, OSError):
        return None


def build_intent_scorer(
    encoder: Encoder | None = None, *, anchors_path: str = "data/intent_anchors.json"
) -> "EmbeddingIntentScorer | None":
    if encoder is None:
        try:
            encoder = Encoder()
            encoder.encode(["probe"])  # force the lazy load
        except Exception:
            return None
    anchors = load_anchors(anchors_path)
    if anchors is not None:
        return EmbeddingIntentScorer(encoder, buy_vectors=anchors[0], browse_vectors=anchors[1])
    return EmbeddingIntentScorer(encoder)
