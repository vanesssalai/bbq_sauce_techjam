from __future__ import annotations

from copilot.contracts import Candidate, SessionState, UserProfile


def distill(state: SessionState) -> str:
    """Short natural-language recap of state.raw_history, cached on state.distilled_summary."""
    if not state.raw_history:
        state.distilled_summary = ""
        return state.distilled_summary
    turns = state.raw_history[-6:]
    parts = [f"{role}: {text}" for role, text in turns]
    summary = " | ".join(parts)
    if len(summary) > 400:
        summary = summary[:400] + "..."
    state.distilled_summary = summary
    return summary


def profile_calib(profile: UserProfile | None, candidate: Candidate) -> float:
    """Small signed adjustment in [-0.05, 0.05] based on how well `candidate` matches
    the user's stated preference_tags and rating_style."""
    if profile is None:
        return 0.0

    score = 0.0
    tags = {t.lower() for t in (profile.preference_tags or [])}
    tokens = {t.lower() for t in candidate.categories} | {t.lower() for t in candidate.colors}
    if candidate.material:
        tokens.add(candidate.material.lower())
    if candidate.brand:
        tokens.add(candidate.brand.lower())

    overlap = len(tags & tokens)
    if overlap:
        score += min(overlap, 3) * 0.02

    if profile.rating_style == "critical" and candidate.average_rating >= 4.5:
        score += 0.02
    elif profile.rating_style == "generous" and candidate.average_rating < 3.5:
        score -= 0.02

    return max(-0.05, min(0.05, score))
