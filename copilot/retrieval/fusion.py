"""Combine the lexical and dense channel lists into one ranking.

Weighted Reciprocal Rank Fusion: fusion consumes **rank position**, not the
score scale, so BM25 scores and cosine similarities never need to be normalized
against each other.
"""

from __future__ import annotations

import os

from copilot.contracts import HARD_FILTER_ATTRS, Query

Channel = list[tuple[str, float]]


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


_W_BM25_FLOOR = _envf("COPILOT_W_BM25_FLOOR", 0.80)
_W_BM25_CEIL = _envf("COPILOT_W_BM25_CEIL", 0.92)


def weights(query: Query) -> dict[str, float]:
    hard = sum(1 for attr in query.slots if attr in HARD_FILTER_ATTRS)
    w_bm25 = _W_BM25_FLOOR + 0.03 * hard + 0.05 * (query.intent_p_buying - 0.5) * 2
    w_bm25 = min(_W_BM25_CEIL, max(_W_BM25_FLOOR, w_bm25))
    return {"bm25": w_bm25, "dense": 1.0 - w_bm25}


def fuse(
    channels: dict[str, Channel],
    channel_weights: dict[str, float],
    *,
    k: int = 60,
    limit: int = 200,
) -> list[tuple[str, float]]:
    accumulated: dict[str, float] = {}
    for name, hits in channels.items():
        weight = channel_weights.get(name, 0.0)
        if weight == 0.0:
            continue
        for rank, (parent_asin, _score) in enumerate(hits, start=1):
            accumulated[parent_asin] = accumulated.get(parent_asin, 0.0) + weight / (k + rank)
    ranked = sorted(accumulated.items(), key=lambda kv: -kv[1])
    return ranked[:limit]
