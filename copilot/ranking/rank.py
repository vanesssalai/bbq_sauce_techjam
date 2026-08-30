from __future__ import annotations

import os

from copilot.contracts import Candidate, Query, RankResult, SessionState


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "on", "true", "yes")


_PASSTHROUGH = _flag("COPILOT_RANK_PASSTHROUGH")

_CE_BLEND = not _flag("COPILOT_CE_BASE")
_CE_BLEND_K = 60
try:
    _CE_BLEND_W = float(os.environ.get("COPILOT_CE_BLEND_W", "1.0"))
except ValueError:
    _CE_BLEND_W = 1.0


def _build_ce_query(q: Query) -> str:
    """Compact render of the Query for the cross-encoder: category · slots · free_text."""
    parts: list[str] = []
    category_slot = q.slots.get("category")
    if category_slot is not None:
        parts.append(category_slot.value)
    for attr, slot in q.slots.items():
        if attr == "category":
            continue
        parts.append(f"{attr}={slot.value}")
    if q.free_text:
        parts.append(q.free_text)
    return " · ".join(parts) if parts else q.free_text


def _ce_doc_text(c: Candidate) -> str:
    text = c.search_text or c.title
    return text


def _ranks_by(candidates: list[Candidate], key) -> dict[str, int]:
    """1-indexed rank of each candidate by descending key value."""
    ordered = sorted(candidates, key=key, reverse=True)
    return {c.parent_asin: i + 1 for i, c in enumerate(ordered)}


def _copeland_tournament(candidates: list[Candidate]) -> list[Candidate]:
    bm25_ranks = _ranks_by(candidates, key=lambda c: c.bm25_score)
    dense_ranks = _ranks_by(candidates, key=lambda c: c.dense_score)
    fused_ranks = {c.parent_asin: (c.fused_rank if c.fused_rank is not None else len(candidates))
                   for c in candidates}
    ce_ranks = _ranks_by(candidates, key=lambda c: c.rank_score if c.rank_score is not None else 0.0)

    def rank_tuple(asin: str) -> tuple[int, int, int, int]:
        return (bm25_ranks[asin], dense_ranks[asin], fused_ranks[asin], ce_ranks[asin])

    copeland_score: dict[str, int] = {c.parent_asin: 0 for c in candidates}

    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            ra, rb = rank_tuple(a.parent_asin), rank_tuple(b.parent_asin)

            a_votes = sum(1 for x, y in zip(ra, rb) if x < y)
            b_votes = sum(1 for x, y in zip(ra, rb) if y < x)
            if a_votes > b_votes:
                copeland_score[a.parent_asin] += 1
                copeland_score[b.parent_asin] -= 1
            elif b_votes > a_votes:
                copeland_score[b.parent_asin] += 1
                copeland_score[a.parent_asin] -= 1

    return sorted(
        candidates,
        key=lambda c: (copeland_score[c.parent_asin], c.rank_score if c.rank_score is not None else 0.0),
        reverse=True,
    )


def rank(
    candidates: list[Candidate],
    q: Query,
    session: SessionState,
    *,
    top_k: int,
    cross_encoder=None,
    weights=None,
    use_tournament: bool = False,
) -> RankResult:
    by_fused = sorted(
        candidates,
        key=lambda c: (c.fused_rank if c.fused_rank is not None else 10**9,
                       -(c.fused_score if c.fused_score is not None else 0.0)),
    )

    if _PASSTHROUGH or cross_encoder is None:
        for i, c in enumerate(by_fused):
            c.rank_score = 1.0 / (1 + i)
        ranked = by_fused[:top_k]
    else:
        try:
            N = int(os.environ.get("COPILOT_CE_TOP_N", "100"))
        except ValueError:
            N = 100
        top_n = by_fused[:N]
        rest = by_fused[N:]

        ce_query = _build_ce_query(q)
        doc_texts = [_ce_doc_text(c) for c in top_n]
        ce_scores = list(cross_encoder.score(ce_query, doc_texts))

        if len(ce_scores) == len(top_n) and top_n:
            if _CE_BLEND:
                order = sorted(range(len(top_n)), key=lambda j: -float(ce_scores[j]))
                ce_rank = {j: pos + 1 for pos, j in enumerate(order)}
                for j, c in enumerate(top_n):
                    fr = c.fused_rank if c.fused_rank is not None else j + 1
                    c.rank_score = (1.0 / (_CE_BLEND_K + fr)
                                    + _CE_BLEND_W / (_CE_BLEND_K + ce_rank[j]))
            else:
                lo, hi = float(min(ce_scores)), float(max(ce_scores))
                span = hi - lo
                for c, raw in zip(top_n, ce_scores):
                    c.rank_score = (float(raw) - lo) / span if span > 0 else 0.5
        else:  # CE failed / wrong length -> fall back to fused order
            for i, c in enumerate(top_n):
                c.rank_score = 1.0 / (1 + i)

        head_lo = min((c.rank_score for c in top_n), default=0.0)
        for i, c in enumerate(rest):
            c.rank_score = head_lo - 1e-3 * (i + 1)
        by_fused = top_n + rest

        if use_tournament:
            M = 20
            by_fused = _copeland_tournament(by_fused[:M]) + by_fused[M:]

    if not (_PASSTHROUGH or cross_encoder is None):
        if use_tournament:
            ranked = by_fused[:top_k]
        else:
            ranked = sorted(by_fused, key=lambda c: c.rank_score, reverse=True)[:top_k]

    if len(ranked) >= 2:
        top = ranked[0].rank_score
        next_few = [c.rank_score for c in ranked[1:4]]
        score_gap = top - (sum(next_few) / len(next_few))
    else:
        score_gap = 0.0

    why = {c.parent_asin: "" for c in ranked[:3]}

    return RankResult(ranked=ranked, score_gap=score_gap, why=why)
