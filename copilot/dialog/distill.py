from __future__ import annotations

import re

from ..contracts import Candidate, UserProfile

_MAX_ABS = 0.10 
_TAG_STEP = 0.02 
_CRITICAL_RATING_DAMP = 0.05
_FIRST_TIMER_PENALTY = 0.03

_LOW_PRIOR_RATING = 3.5
_FIRST_TIMER_HINTS = ("first", "new to", "rarely", "occasional", "infreq", "0 prior", "1 prior", "2 prior")

_WORD = re.compile(r"[a-z0-9]+")


def _tag_hits(candidate: Candidate, tags: list[str]) -> int:
    hay = set(_WORD.findall(f"{candidate.title} {candidate.search_text}".lower()))
    return sum(1 for t in tags if t.lower() in hay)


def profile_calib(candidate: Candidate, profile: UserProfile | None) -> float:
    if profile is None:
        return 0.0

    nudge = 0.0

    tags = list(profile.preference_tags or [])
    hits = _tag_hits(candidate, tags) if tags else 0
    nudge += _TAG_STEP * min(hits, 3)

    critical = (profile.rating_style or "").strip().lower() == "critical"
    low_bar = bool(profile.average_prior_rating) and profile.average_prior_rating < _LOW_PRIOR_RATING
    if critical or low_bar:
        nudge -= _CRITICAL_RATING_DAMP * (candidate.average_rating / 5.0)

    freq = (profile.purchase_frequency or "").strip().lower()
    if any(h in freq for h in _FIRST_TIMER_HINTS) and hits == 0:
        nudge -= _FIRST_TIMER_PENALTY

    return max(-_MAX_ABS, min(_MAX_ABS, nudge))


def distill(state: SessionState) -> str:
    """Short natural-language recap of `state.raw_history` (stored back on
    `state.distilled_summary`), so later turns keep the prompt small."""
    raise NotImplementedError
