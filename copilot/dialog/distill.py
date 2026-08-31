"""Profile calibration for the reranker (RETRIEVAL_RERANK_BUILD_GUIDE.md §4b).

`profile_calib(candidate, profile)` turns the anonymized aggregate `user_profile`
into a small signed adjustment that `ranking.rank._soft_adjustments` (step 7c)
adds to `candidate.rank_score`. It never dominates the retrieval / cross-encoder
signal -- the return is clamped to a few hundredths.
"""

from __future__ import annotations

import re

from copilot.contracts import Candidate, UserProfile

_TAG_STEP = 0.02             # per matched preference tag, capped at 3
_FIRST_TIMER_PENALTY = 0.03  # first-time / infrequent buyer + a loose match
_HIGH_BAR_RATING_DAMP = 0.04  # critical raters: shrink the raw-rating bonus
_HIGH_BAR_MATCH_BONUS = 0.03  # ...and instead reward a tight slot match
_CLAMP = 0.06

_WORD_RE = re.compile(r"[a-z0-9]+")
_FIRST_TIMER_RE = re.compile(
    r"first[- ]?time|new to|no prior|rarely|infrequent|occasional|one[- ]?off", re.I
)


def _is_first_timer(purchase_frequency: str) -> bool:
    pf = (purchase_frequency or "").lower()
    if _FIRST_TIMER_RE.search(pf):
        return True
    m = re.search(r"(\d+)", pf)          # "0 prior purchases", "1 prior purchase"
    return m is not None and int(m.group(1)) <= 1


def _has_high_bar(profile: UserProfile) -> bool:
    style = (profile.rating_style or "").lower()
    if any(k in style for k in ("critic", "harsh", "demanding", "picky")):
        return True
    return 0.0 < profile.average_prior_rating < 3.0


def profile_calib(
    candidate: Candidate,
    profile: UserProfile | None,
    *,
    slot_match: float = 0.0,
) -> float:
    """Signed nudge for `candidate.rank_score`, clamped to +/- 0.06. 0.0 when no
    profile. `slot_match` in [0, 1] is how tightly the candidate matched the
    stated query slots (0 = caller did not compute it).

    - `preference_tags` present in the candidate's `search_text`  -> small +
    - first-time / infrequent buyer on a loose match             -> small -
    - critical `rating_style` / low `average_prior_rating`        -> dampen the
      raw-rating bonus, add back a reward for a tight slot match
    """
    if profile is None:
        return 0.0

    score = 0.0
    sm = max(0.0, min(1.0, slot_match))

    tags = [t.lower() for t in (profile.preference_tags or []) if t]
    if tags:
        hay = set(_WORD_RE.findall((candidate.search_text or "").lower()))
        score += _TAG_STEP * min(sum(1 for t in tags if t in hay), 3)

    if _is_first_timer(profile.purchase_frequency) and sm < 0.5:
        score -= _FIRST_TIMER_PENALTY

    if _has_high_bar(profile):
        score -= _HIGH_BAR_RATING_DAMP * (candidate.average_rating / 5.0)
        score += _HIGH_BAR_MATCH_BONUS * sm

    return max(-_CLAMP, min(_CLAMP, score))
