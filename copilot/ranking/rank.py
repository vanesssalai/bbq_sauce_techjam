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
    """Compact render of the Query for the cross-encoder: category \u00b7 slots \u00b7 free_text."""
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
    return " \u00b7 ".join(parts) if parts else q.free_text


def _ce_doc_text(c: Candidate) -> str:
    return c.search_text or c.title


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


def _soft_adjustments(
    candidates: list[Candidate],
    q: Query,
    session: SessionState,
    weights: dict[str, float] | None,
) -> None:
    """In-place additive nudges to c.rank_score: price tier, category match,
    tag overlap (color/material/brand), negation penalty, rating prior, profile calibration."""
    from copilot.dialog.distill import profile_calib

    w = {
        "price_tier": 0.05,
        "category_match": 0.05,
        "tag_overlap": 0.03,
        "negation_penalty": 0.20,
        "rating_prior": 0.03,
        "profile_calib": 1.0,
    }
    if weights:
        w.update(weights)

    price_min_slot = q.slots.get("price_min")
    price_max_slot = q.slots.get("price_max")
    category_slot = q.slots.get("category")
    soft_slots = {attr: q.slots[attr] for attr in ("color", "material", "brand") if attr in q.slots}

    for c in candidates:
        adj = 0.0

        if c.price is not None:
            try:
                if price_min_slot is not None and c.price < float(price_min_slot.value):
                    adj -= w["price_tier"]
                elif price_max_slot is not None and c.price > float(price_max_slot.value):
                    adj -= w["price_tier"]
                elif price_min_slot is not None or price_max_slot is not None:
                    adj += w["price_tier"]
            except (TypeError, ValueError):
                pass

        if category_slot is not None and c.categories:
            if category_slot.value.lower() in [cat.lower() for cat in c.categories]:
                adj += w["category_match"]

        overlap = 0
        for attr, slot in soft_slots.items():
            if attr == "color" and slot.value.lower() in [col.lower() for col in c.colors]:
                overlap += 1
            elif attr == "material" and c.material and slot.value.lower() == c.material.lower():
                overlap += 1
            elif attr == "brand" and c.brand and slot.value.lower() == c.brand.lower():
                overlap += 1
        if overlap:
            adj += w["tag_overlap"] * overlap

        for attr, bad_values in (q.negations or {}).items():
            bad_lower = {v.lower() for v in bad_values}
            candidate_val = None
            if attr == "color":
                candidate_val = [col.lower() for col in c.colors]
            elif attr == "material" and c.material:
                candidate_val = [c.material.lower()]
            elif attr == "brand" and c.brand:
                candidate_val = [c.brand.lower()]
            elif attr == "category":
                candidate_val = [cat.lower() for cat in c.categories]
            if candidate_val and bad_lower & set(candidate_val):
                adj -= w["negation_penalty"]

        adj += w["rating_prior"] * (c.average_rating / 5.0)
        adj += w["profile_calib"] * profile_calib(session.user_profile, c)

        c.rank_score = (c.rank_score if c.rank_score is not None else 0.0) + adj


def _similarity(a: Candidate, b: Candidate) -> float:
    """Attribute-overlap similarity in [0, 1] (categories/material/color/brand) -- a
    lightweight stand-in for dense-embedding cosine sim, since rank() isn't passed vectors."""
    a_tok = {t.lower() for t in a.categories} | {t.lower() for t in a.colors}
    b_tok = {t.lower() for t in b.categories} | {t.lower() for t in b.colors}
    if a.material:
        a_tok.add(a.material.lower())
    if b.material:
        b_tok.add(b.material.lower())
    if a.brand:
        a_tok.add(a.brand.lower())
    if b.brand:
        b_tok.add(b.brand.lower())
    if not a_tok or not b_tok:
        return 0.0
    inter = len(a_tok & b_tok)
    union = len(a_tok | b_tok)
    return inter / union if union else 0.0


def _mmr_diversify(candidates: list[Candidate], top_k: int, lam: float = 0.7) -> list[Candidate]:
    """Greedy MMR: pick argmax(lam*relevance - (1-lam)*max_sim_to_selected)."""
    if not candidates:
        return []
    pool = list(candidates)
    selected: list[Candidate] = [pool.pop(0)]
    while pool and len(selected) < top_k:
        best_idx, best_score = 0, float("-inf")
        for i, c in enumerate(pool):
            relevance = c.rank_score if c.rank_score is not None else 0.0
            max_sim = max((_similarity(c, s) for s in selected), default=0.0)
            mmr_score = lam * relevance - (1 - lam) * max_sim
            if mmr_score > best_score:
                best_idx, best_score = i, mmr_score
        selected.append(pool.pop(best_idx))
    return selected


def _build_why(candidates: list[Candidate], q: Query, n: int = 3) -> dict[str, str]:
    why: dict[str, str] = {}
    category_slot = q.slots.get("category")
    soft_slots = {attr: q.slots[attr] for attr in ("color", "material", "brand") if attr in q.slots}
    for c in candidates[:n]:
        reasons: list[str] = []
        if category_slot is not None and category_slot.value.lower() in [cat.lower() for cat in c.categories]:
            reasons.append(f"matches category \'{category_slot.value}\'")
        for attr, slot in soft_slots.items():
            if attr == "color" and slot.value.lower() in [col.lower() for col in c.colors]:
                reasons.append(f"color \'{slot.value}\'")
            elif attr == "material" and c.material and slot.value.lower() == c.material.lower():
                reasons.append(f"material \'{slot.value}\'")
            elif attr == "brand" and c.brand and slot.value.lower() == c.brand.lower():
                reasons.append(f"brand \'{slot.value}\'")
        if c.average_rating >= 4.5:
            reasons.append("highly rated")
        why[c.parent_asin] = "; ".join(reasons) if reasons else ""
    return why


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
    else:
        try:
            N = int(os.environ.get("COPILOT_CE_TOP_N", "30"))
        except ValueError:
            N = 30
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
        else:
            for i, c in enumerate(top_n):
                c.rank_score = 1.0 / (1 + i)

        head_lo = min((c.rank_score for c in top_n), default=0.0)
        for i, c in enumerate(rest):
            c.rank_score = head_lo - 1e-3 * (i + 1)
        by_fused = top_n + rest

        if use_tournament:
            M = 20
            by_fused = _copeland_tournament(by_fused[:M]) + by_fused[M:]
            # Re-stamp rank_score to reflect the tournament\'s decided order positionally,
            # so soft adjustments below nudge *within* that order rather than discarding it.
            for i, c in enumerate(by_fused):
                c.rank_score = 1.0 / (1 + i)

    # 7c: soft score adjustments -- small additive nudges on top of the base score.
    _soft_adjustments(by_fused, q, session, weights)

    pool = sorted(by_fused, key=lambda c: c.rank_score, reverse=True)
    if q.track == "browsing":
        # 7d: MMR diversity, browsing track only.
        window = pool[:max(top_k * 3, 30)]
        ranked = _mmr_diversify(window, top_k)
    else:
        ranked = pool[:top_k]

    if len(ranked) >= 2:
        top = ranked[0].rank_score
        next_few = [c.rank_score for c in ranked[1:4]]
        score_gap = top - (sum(next_few) / len(next_few))
    else:
        score_gap = 0.0

    why = _build_why(ranked, q)

    return RankResult(ranked=ranked, score_gap=score_gap, why=why)
