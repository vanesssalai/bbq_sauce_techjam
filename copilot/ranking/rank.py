from __future__ import annotations

from copilot.contracts import Candidate, Query, RankResult, SessionState


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
    """
    Copeland rank-aggregation over {bm25_rank, dense_rank, fused_rank, ce_rank}.
    For each pair (a, b): a wins iff a majority of the four rankings place a above b.
    copeland(a) = wins - losses. Re-sort by copeland score, ties broken by ce (rank_score).
    """
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
            # lower rank number = better; count how many of the 4 rankings prefer a over b
            a_votes = sum(1 for x, y in zip(ra, rb) if x < y)
            b_votes = sum(1 for x, y in zip(ra, rb) if y < x)
            if a_votes > b_votes:
                copeland_score[a.parent_asin] += 1
                copeland_score[b.parent_asin] -= 1
            elif b_votes > a_votes:
                copeland_score[b.parent_asin] += 1
                copeland_score[a.parent_asin] -= 1
            # tie in votes -> no change to either

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
    # Work on a stable order by fused_rank/fused_score first.
    by_fused = sorted(
        candidates,
        key=lambda c: c.fused_score if c.fused_score is not None else 0.0,
        reverse=True,
    )

    if cross_encoder is None:
        # Degraded mode: no cross-encoder available, base = fused_score.
        for c in by_fused:
            c.rank_score = c.fused_score if c.fused_score is not None else 0.0
    else:
        # 7a: cross-encoder rerank over the top N by fused_rank.
        N = 100
        top_n = by_fused[:N]
        rest = by_fused[N:]

        ce_query = _build_ce_query(q)
        doc_texts = [_ce_doc_text(c) for c in top_n]
        ce_scores = cross_encoder.score(ce_query, doc_texts)

        # min-max normalize over N -> base in [0, 1]
        if len(ce_scores) > 0:
            lo, hi = float(min(ce_scores)), float(max(ce_scores))
            span = hi - lo
            for c, raw in zip(top_n, ce_scores):
                c.rank_score = (float(raw) - lo) / span if span > 0 else 0.5
        for c in rest:
            c.rank_score = c.fused_score if c.fused_score is not None else 0.0

        by_fused = top_n + rest

        # 7b: tournament head over the top M=20, only if a cross-encoder ran.
        if use_tournament:
            M = 20
            top_m = by_fused[:M]
            rest_after_m = by_fused[M:]
            reordered_top_m = _copeland_tournament(top_m)
            by_fused = reordered_top_m + rest_after_m

    if use_tournament and cross_encoder is not None:
        # Preserve the tournament's Copeland order for the top M; rest stays fused/ce order.
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
