from __future__ import annotations

from ..contracts import Candidate, Query, RankResult, SessionState

_CE_TOP_N = 100
_CE_DOC_MAX_TOKENS = 60 

_BELOW_BAND = -1.0 

_TOURNAMENT_TOP_M = 20
_BT_ITERS = 60   

def _track(q: Query) -> str:
    return "buying" if q.intent_p_buying >= 0.5 else "browsing"


def _build_ce_query(q: Query) -> str:
    parts: list[str] = []
    if q.category_anchor:
        parts.append(q.category_anchor)
    for attr, slot in (*q.hard_slots.items(), *q.soft_slots.items()):
        parts.append(f"{attr}={slot.value}")
    if q.free_text:
        parts.append(q.free_text)
    return " · ".join(parts) if parts else q.free_text


def _ce_doc_text(c: Candidate) -> str:
    text = (c.search_text or c.title or "").strip()
    tokens = text.split()
    if len(tokens) > _CE_DOC_MAX_TOKENS:
        return " ".join(tokens[:_CE_DOC_MAX_TOKENS])
    return text


def _ranks_by(candidates: list[Candidate], key) -> dict[str, int]:
    """1-indexed rank of each candidate by descending key value."""
    ordered = sorted(candidates, key=key, reverse=True)
    return {c.parent_asin: i + 1 for i, c in enumerate(ordered)}


def _by_fused(candidates: list[Candidate]) -> list[Candidate]:
    incoming = {id(c): i for i, c in enumerate(candidates)}
    return sorted(
        candidates,
        key=lambda c: (
            c.fused_rank if c.fused_rank is not None else 10**9,
            -(c.fused_score if c.fused_score is not None else 0.0),
            incoming[id(c)],
        ),
    )


def _active_rankings(candidates: list[Candidate]) -> list[dict[str, int]]:
    rankings: list[dict[str, int]] = []
    if len({c.bm25_score for c in candidates}) > 1:
        rankings.append(_ranks_by(candidates, key=lambda c: c.bm25_score))
    if len({c.dense_score for c in candidates}) > 1:
        rankings.append(_ranks_by(candidates, key=lambda c: c.dense_score))
    if len({c.fused_rank for c in candidates if c.fused_rank is not None}) > 1:
        rankings.append(_ranks_by(
            candidates,
            key=lambda c: c.fused_rank if c.fused_rank is not None else 10**9,
            reverse=False,
        ))
    if len({c.rank_score for c in candidates if c.rank_score is not None}) > 1:
        rankings.append(_ranks_by(
            candidates,
            key=lambda c: c.rank_score if c.rank_score is not None else _BELOW_BAND,
        ))
    return rankings


def _copeland_tournament(candidates: list[Candidate]) -> list[Candidate]:
    """Copeland rank-aggregation (handoff §6, ranking method ③). Raw CE logits
    are poorly calibrated, so aggregate the orderings already in hand
    ({bm25, dense, fused, ce} ranks that carry signal). Each pair (a, b) is a
    pairwise-majority contest -- `a` wins it when more of the active rankings put
    `a` above `b`; `copeland(a) = wins − losses`. Re-sort by that, ties broken by
    the cross-encoder score. Zero extra model calls."""
    rankings = _active_rankings(candidates)
    if not rankings:
        return list(candidates)

    score: dict[str, int] = {c.parent_asin: 0 for c in candidates}
    for i, a in enumerate(candidates):
        ai = a.parent_asin
        for b in candidates[i + 1:]:
            bi = b.parent_asin
            a_votes = sum(1 for r in rankings if r[ai] < r[bi])
            b_votes = sum(1 for r in rankings if r[bi] < r[ai])
            if a_votes > b_votes:
                score[ai] += 1
                score[bi] -= 1
            elif b_votes > a_votes:
                score[bi] += 1
                score[ai] -= 1

    return sorted(
        candidates,
        key=lambda c: (score[c.parent_asin], c.rank_score if c.rank_score is not None else 0.0),
        reverse=True,
    )


def _perturbed_ce_queries(q: Query) -> list[str]:
    anchor = q.category_anchor
    slots = [f"{a}={s.value}" for a, s in (*q.hard_slots.items(), *q.soft_slots.items())]
    ft = q.free_text
    raw = [
        _build_ce_query(q),
        " · ".join(p for p in (anchor, ft) if p),
        " · ".join(p for p in (anchor, *slots) if p),
        ft or anchor,
        " · ".join(p for p in (anchor, *reversed(slots), ft) if p),
    ]
    out: list[str] = []
    for v in (s.strip() for s in raw):
        if v and v not in out:
            out.append(v)
    return out or [q.free_text or q.category_anchor or ""]


def _bradley_terry(candidates: list[Candidate], q: Query, cross_encoder) -> list[Candidate]:
    asins = [c.parent_asin for c in candidates]
    docs = [_ce_doc_text(c) for c in candidates]
    wins: dict[str, dict[str, float]] = {a: {b: 0.0 for b in asins} for a in asins}

    for qv in _perturbed_ce_queries(q):
        try:
            sc = [float(x) for x in cross_encoder.score(qv, docs)]
        except Exception:
            continue
        if len(sc) != len(asins):
            continue
        for i in range(len(asins)):
            for j in range(i + 1, len(asins)):
                if sc[i] > sc[j]:
                    wins[asins[i]][asins[j]] += 1.0
                elif sc[j] > sc[i]:
                    wins[asins[j]][asins[i]] += 1.0
                else:
                    wins[asins[i]][asins[j]] += 0.5
                    wins[asins[j]][asins[i]] += 0.5

    if not any(wins[a][b] for a in asins for b in asins if a != b):
        return list(candidates)

    strength = {a: 1.0 for a in asins}
    for _ in range(_BT_ITERS):
        nxt: dict[str, float] = {}
        for a in asins:
            num = sum(wins[a][b] for b in asins if b != a)
            den = sum(
                (wins[a][b] + wins[b][a]) / (strength[a] + strength[b])
                for b in asins if b != a
            )
            nxt[a] = num / den if den > 0 else strength[a]
        mean = sum(nxt.values()) / len(nxt) or 1.0
        strength = {a: v / mean for a, v in nxt.items()}

    return sorted(candidates, key=lambda c: strength[c.parent_asin], reverse=True)


def rank(
    candidates: list[Candidate],
    q: Query,
    session: SessionState,
    *,
    top_k: int,
    cross_encoder=None,
    weights=None,
    use_tournament: bool = False,
    tournament_method: str = "copeland",
) -> RankResult:
    ordered = _by_fused(candidates)

    ce_ran = False
    if cross_encoder is not None and ordered:
        head, tail = ordered[:_CE_TOP_N], ordered[_CE_TOP_N:]
        try:
            raw = cross_encoder.score(_build_ce_query(q), [_ce_doc_text(c) for c in head])
            ce_scores = [float(x) for x in raw]
        except Exception:
            ce_scores = []
        if len(ce_scores) == len(head) and head:
            lo, hi = min(ce_scores), max(ce_scores)
            span = hi - lo
            for c, s in zip(head, ce_scores):
                c.rank_score = (s - lo) / span if span > 0 else 0.5

            for i, c in enumerate(tail):
                c.rank_score = _BELOW_BAND - 1.0 - i
            ordered = head + tail
            ce_ran = True

    if not ce_ran:
        for i, c in enumerate(ordered):
            c.rank_score = c.fused_score if c.fused_score is not None else _BELOW_BAND - i

    if use_tournament and ce_ran:
        m = min(_TOURNAMENT_TOP_M, len(ordered))
        head_m = ordered[:m]
        if tournament_method == "bradley_terry":
            head_m = _bradley_terry(head_m, q, cross_encoder)
        else:
            head_m = _copeland_tournament(head_m)
        ordered = head_m + ordered[m:]
        ranked = ordered[:top_k]
    else:
        ranked = sorted(ordered, key=lambda c: c.rank_score, reverse=True)[:top_k]

    if len(ranked) >= 2:
        nxt = [c.rank_score for c in ranked[1:4]]
        score_gap = ranked[0].rank_score - (sum(nxt) / len(nxt))
    else:
        score_gap = 0.0

    why = {c.parent_asin: "" for c in ranked[:3]}

    return RankResult(ranked=ranked, score_gap=score_gap, why=why)
