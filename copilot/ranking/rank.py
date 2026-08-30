from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path

from ..contracts import Candidate, Query, RankResult, SessionState, UserProfile
from ..dialog.distill import profile_calib
from ..retrieval.filters import _violates_negation  # reuse: colour/material/brand exclusion

_CE_TOP_N = 100
_CE_DOC_MAX_TOKENS = 60

_BELOW_BAND = -1.0

_TOURNAMENT_TOP_M = 20
_BT_ITERS = 60

_PRICE_OVER_HARD = 2.0 
_NEGATION_PENALTY = 5.0 
_RATING_SATURATION = math.log1p(3000) 
_WORD_RE = re.compile(r"[a-z0-9]+")

_MMR_LAMBDA = 0.70 
_MMR_POOL = 60
_MMR_MIN_POOL = 4 


@dataclass
class RankWeights:
    price_tier: float = 0.15
    category: float = 0.12
    soft_slot: float = 0.08
    tag_overlap: float = 0.05
    rating: float = 0.05


_WEIGHTS_PATH = Path(__file__).with_name("ranker_weights.json")
_DEFAULT_WEIGHTS: RankWeights | None = None


def load_rank_weights(path: Path | str | None = None) -> RankWeights:
    p = Path(path) if path is not None else _WEIGHTS_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(RankWeights)}
        return RankWeights(**{k: float(v) for k, v in data.items() if k in allowed})
    except Exception:
        return RankWeights()


def _default_weights() -> RankWeights:
    global _DEFAULT_WEIGHTS
    if _DEFAULT_WEIGHTS is None:
        _DEFAULT_WEIGHTS = load_rank_weights()
    return _DEFAULT_WEIGHTS


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


def _ranks_by(candidates: list[Candidate], key, *, reverse: bool = True) -> dict[str, int]:
    """1-indexed rank of each candidate by `key` (default: descending = higher better)."""
    ordered = sorted(candidates, key=key, reverse=reverse)
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


def _num(slot) -> float | None:
    try:
        return float(slot.value)
    except (TypeError, ValueError, AttributeError):
        return None


def _price_tier_bonus(c: Candidate, q: Query) -> float:
    if c.price is None:
        return 0.0
    lo = _num(q.hard_slots.get("price_min"))
    hi = _num(q.hard_slots.get("price_max"))
    if hi is not None and hi > 0:
        if c.price > hi:
            over = (c.price - hi) / (hi * (_PRICE_OVER_HARD - 1.0))
            return -min(1.0, over)
    if lo is not None and lo > 0 and c.price < lo:
        return -min(1.0, (lo - c.price) / lo)
    return 1.0 if (hi is not None or lo is not None) else 0.0


def _category_match(c: Candidate, q: Query) -> float:
    hay = " ".join(c.categories).lower()
    dept = (c.department or "").lower()
    probes = [q.category_anchor] + [
        q.hard_slots[k].value for k in ("category", "department") if k in q.hard_slots
    ]
    for probe in (str(p).lower().strip() for p in probes if p):
        toks = [t for t in probe.split() if len(t) > 2]
        if hay and (probe in hay or (toks and all(t in hay for t in toks))):
            return 1.0
        if dept and (probe == dept or probe in dept or dept in probe):
            return 1.0
    return 0.0


def _soft_slot_match(c: Candidate, q: Query) -> float:
    if not q.soft_slots:
        return 0.0
    hits = 0
    for attr, slot in q.soft_slots.items():
        v = str(slot.value).lower()
        if attr == "color":
            if any(v == x.lower() or v in x.lower() or x.lower() in v for x in c.colors):
                hits += 1
        elif attr == "material" and c.material and (v in c.material.lower() or c.material.lower() in v):
            hits += 1
        elif attr == "brand" and c.brand and (v in c.brand.lower() or c.brand.lower() in v):
            hits += 1
    return hits / len(q.soft_slots)


def _tag_overlap(c: Candidate, profile: UserProfile | None) -> float:
    if profile is None or not profile.preference_tags:
        return 0.0
    hay = set(_WORD_RE.findall(f"{c.title} {c.search_text}".lower()))
    hits = sum(1 for t in profile.preference_tags if t.lower() in hay)
    return min(hits, 3) / 3.0


def _rating_prior(c: Candidate) -> float:
    if c.rating_number <= 0 or c.average_rating <= 0:
        return 0.0
    volume = min(1.0, math.log1p(c.rating_number) / _RATING_SATURATION)
    return volume * (c.average_rating / 5.0)


def _soft_adjustment(c: Candidate, q: Query, profile: UserProfile | None, w: RankWeights) -> float:
    adj = (
        w.price_tier * _price_tier_bonus(c, q)
        + w.category * _category_match(c, q)
        + w.soft_slot * _soft_slot_match(c, q)
        + w.tag_overlap * _tag_overlap(c, profile)
        + w.rating * _rating_prior(c)
        + profile_calib(c, profile)
    )
    if q.negations and _violates_negation(c, q.negations):
        adj -= _NEGATION_PENALTY
    return adj


def _signature(c: Candidate) -> set[str]:
    sig: set[str] = set()
    for cat in c.categories:
        sig.update(f"c:{t}" for t in _WORD_RE.findall(cat.lower()) if len(t) > 2)
    sig.update(f"col:{x.lower()}" for x in c.colors)
    if c.material:
        sig.add(f"m:{c.material.lower()}")
    if c.brand:
        sig.add(f"b:{c.brand.lower()}")
    if c.department:
        sig.add(f"d:{c.department.lower()}")
    sig.update(f"t:{t}" for t in _WORD_RE.findall(c.title.lower())[:8] if len(t) > 2)
    return sig


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _mmr_order(ranked_desc: list[Candidate], top_k: int, *, lam: float = _MMR_LAMBDA) -> list[Candidate]:
    pool = ranked_desc[:_MMR_POOL]
    if len(pool) < _MMR_MIN_POOL or top_k <= 1:
        return ranked_desc[:top_k]

    rels = [max(0.0, c.rank_score if c.rank_score is not None else 0.0) for c in pool]
    hi = max(rels) or 1.0
    rel_norm = {id(c): r / hi for c, r in zip(pool, rels)}
    sigs = {id(c): _signature(c) for c in pool}

    selected = [pool[0]]
    chosen = {id(pool[0])}
    while len(selected) < min(top_k, len(pool)):
        best, best_mmr = None, None
        for c in pool:
            if id(c) in chosen:
                continue
            div = max(_jaccard(sigs[id(c)], sigs[id(s)]) for s in selected)
            mmr = lam * rel_norm[id(c)] - (1.0 - lam) * div
            if best_mmr is None or mmr > best_mmr:
                best, best_mmr = c, mmr
        selected.append(best)
        chosen.add(id(best))

    if len(selected) < top_k:
        selected.extend(ranked_desc[len(pool):top_k])
    return selected


def rank(
    candidates: list[Candidate],
    q: Query,
    session: SessionState,
    *,
    top_k: int,
    cross_encoder=None,
    weights: RankWeights | None = None,
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

    if use_tournament and ce_ran and len(ordered) > 1:
        m = min(_TOURNAMENT_TOP_M, len(ordered))
        head_m, tail_m = ordered[:m], ordered[m:]
        head_m = (_bradley_terry(head_m, q, cross_encoder)
                  if tournament_method == "bradley_terry"
                  else _copeland_tournament(head_m))

        head_scores = sorted((c.rank_score or 0.0 for c in ordered[:m]), reverse=True)
        for pos, c in enumerate(head_m):
            c.rank_score = head_scores[pos]
        floor = head_scores[-1] if head_scores else 0.0
        for i, c in enumerate(tail_m):
            c.rank_score = floor - 1e-6 * (i + 1)
        ordered = head_m + tail_m

    w = weights if weights is not None else _default_weights()
    profile = session.user_profile if session is not None else None
    for c in ordered:
        base = c.rank_score if c.rank_score is not None else 0.0
        c.rank_score = base + _soft_adjustment(c, q, profile, w)

    by_relevance = sorted(ordered, key=lambda c: c.rank_score, reverse=True)

    if _track(q) == "browsing":
        ranked = _mmr_order(by_relevance, top_k)
    else:
        ranked = by_relevance[:top_k]

    if len(ranked) >= 2:
        nxt = [c.rank_score for c in ranked[1:4]]
        score_gap = ranked[0].rank_score - (sum(nxt) / len(nxt))
    else:
        score_gap = 0.0

    why = {c.parent_asin: "" for c in ranked[:3]}

    return RankResult(ranked=ranked, score_gap=score_gap, why=why)
