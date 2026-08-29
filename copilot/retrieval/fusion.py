"""Combine the lexical and dense channel lists into one ranking.

Weighted Reciprocal Rank Fusion: fusion consumes **rank position**, not the
score scale, so BM25 scores and cosine similarities never need to be normalized
against each other.
"""

from __future__ import annotations

from copilot.contracts import HARD_FILTER_ATTRS, Query

Channel = list[tuple[str, float]]


def weights(query: Query) -> dict[str, float]:
    """Per-turn blend weights. More hard slots or a stronger buying intent pushes
    weight toward the lexical channel; browsing / vague turns lean dense."""
    hard = sum(1 for attr in query.slots if attr in HARD_FILTER_ATTRS)
    w_bm25 = 0.35 + 0.12 * hard + 0.25 * (query.intent_p_buying - 0.5) * 2
    w_bm25 = min(0.80, max(0.30, w_bm25))
    return {"bm25": w_bm25, "dense": 1.0 - w_bm25}


def fuse(
    channels: dict[str, Channel],
    channel_weights: dict[str, float],
    *,
    k: int = 60,
    limit: int = 200,
) -> list[tuple[str, float]]:
    """Merge `{channel_name: [(parent_asin, score), ...]}` (each best-first) into
    one best-first `[(parent_asin, fused_score)]`, length <= `limit`.

        fused[d] = Σ_channel  weight[c] / (k + rank_c(d))     # rank_c 1-indexed

    A document absent from a channel contributes 0 for that channel.
    """
    accumulated: dict[str, float] = {}
    for name, hits in channels.items():
        weight = channel_weights.get(name, 0.0)
        if weight == 0.0:
            continue
        for rank, (parent_asin, _score) in enumerate(hits, start=1):
            accumulated[parent_asin] = accumulated.get(parent_asin, 0.0) + weight / (k + rank)
    ranked = sorted(accumulated.items(), key=lambda kv: -kv[1])
    return ranked[:limit]
